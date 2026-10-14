#!/usr/bin/env bash
# Read-only checks before launching anything:
# account and region, whether both instance types are offered (and where else they
# are if not), the on-demand vCPU quota, the Ubuntu AMIs, current prices and the
# cost estimate, then `provision.sh --dry-run`. Creates nothing.
#
#   cloud/preflight.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_aws
ARM=$(instance_type arm64)
X86=$(instance_type x86_64)

# Stage durations per machine, from docs/EXPERIMENT_PLAN.md (hours, incl. ~0.5 h setup).
PILOT_224_HOURS=6.3
FULL_HOURS=18.5

echo "AWS CLI: $(aws --version 2>&1)"
aws sts get-caller-identity --query '[Account, Arn]' --output text |
    awk '{ n = split($2, a, ":"); printf "account: ********%s  principal: %s\n", substr($1, length($1) - 3), a[n] }'
echo "region: $REGION"

log "instance type offerings in $REGION"
offered=$(aws ec2 describe-instance-type-offerings --location-type region \
    --filters "Name=instance-type,Values=$ARM,$X86" --query 'InstanceTypeOfferings[].InstanceType' --output text)
for t in $ARM $X86; do
    if echo "$offered" | grep -qw "$t"; then
        azs=$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
            --filters "Name=instance-type,Values=$t" --query 'InstanceTypeOfferings[].Location' --output text)
        echo "    $t: offered (AZs: $azs)"
    else
        echo "    $t: NOT offered in $REGION"
    fi
done
if ! { echo "$offered" | grep -qw "$ARM" && echo "$offered" | grep -qw "$X86"; }; then
    log "regions offering both $ARM and $X86:"
    for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
        n=$(aws ec2 describe-instance-type-offerings --region "$r" --location-type region \
            --filters "Name=instance-type,Values=$ARM,$X86" --query 'length(InstanceTypeOfferings)' --output text 2>/dev/null || echo 0)
        [ "$n" = 2 ] && echo "    $r"
    done
fi

log "on-demand vCPU quota (Running On-Demand Standard instances, L-1216C47A)"
quota=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text 2>/dev/null ||
    aws service-quotas get-aws-default-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text)
used=0
running_types=$(aws ec2 describe-instances --filters Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].InstanceType' --output text)
for t in $running_types; do
    case $t in a* | c* | d* | h* | i* | m* | r* | t* | z*) ;; *) continue ;; esac
    v=$(aws ec2 describe-instance-types --instance-types "$t" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
    used=$((used + v))
done
need=$(( $(aws ec2 describe-instance-types --instance-types "$ARM" "$X86" --query 'sum(InstanceTypes[].VCpuInfo.DefaultVCpus)' --output text) ))
echo "    quota ${quota%.*} vCPUs, in use $used, needed $need -> $([ $((used + need)) -le "${quota%.*}" ] && echo OK || echo 'NOT ENOUGH: request an increase')"

log "Ubuntu 24.04 LTS AMIs (SSM public parameters)"
for arch in arm64 x86_64; do
    ami=$(aws ssm get-parameter --name "$(ami_parameter "$arch")" --query Parameter.Value --output text)
    echo "    $arch: $ami $(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].[Name,CreationDate]' --output text)"
done

log "on-demand prices in $REGION (AWS Pricing API) and cost estimate"
price() {  # $1 instance type -> USD per hour
    aws pricing get-products --region us-east-1 --service-code AmazonEC2 --output json --filters \
        "Type=TERM_MATCH,Field=instanceType,Value=$1" "Type=TERM_MATCH,Field=regionCode,Value=$REGION" \
        "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
        "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" "Type=TERM_MATCH,Field=capacitystatus,Value=Used" |
        python3 -c '
import json, sys
for item in json.load(sys.stdin)["PriceList"]:
    for term in json.loads(item)["terms"]["OnDemand"].values():
        for dim in term["priceDimensions"].values():
            print(dim["pricePerUnit"]["USD"]); sys.exit()
print("nan")'
}
gp3=$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 --output json --filters \
    "Type=TERM_MATCH,Field=volumeApiName,Value=gp3" "Type=TERM_MATCH,Field=regionCode,Value=$REGION" \
    "Type=TERM_MATCH,Field=productFamily,Value=Storage" |
    python3 -c '
import json, sys
for item in json.load(sys.stdin)["PriceList"]:
    for term in json.loads(item)["terms"]["OnDemand"].values():
        for dim in term["priceDimensions"].values():
            print(dim["pricePerUnit"]["USD"]); sys.exit()
print("nan")')
p_arm=$(price "$ARM")
p_x86=$(price "$X86")
python3 - "$p_arm" "$p_x86" "$gp3" "$VOLUME_GB" "$PILOT_224_HOURS" "$FULL_HOURS" "$SHUTDOWN_HOURS" "$ARM" "$X86" <<'EOF'
import sys
p_arm, p_x86, gp3, vol, h1, h2, deadline = map(float, sys.argv[1:8])
arm, x86 = sys.argv[8:10]
ebs_per_h = 2 * vol * gp3 / 730  # two root volumes, gp3 billed per GB-month
ipv4_per_h = 2 * 0.005           # public IPv4 addresses, $0.005/h each (AWS list price)
print(f"    {arm}: ${p_arm:.4f}/h   {x86}: ${p_x86:.4f}/h   gp3: ${gp3:.3f}/GB-month ({2 * vol:.0f} GB total)"
      f"   public IPv4: 2 x $0.005/h")
for name, h in (("pilot + 224 stage", h1), ("full run (pilot + 224 + 448)", h2),
                (f"worst case: both forgotten until the {deadline:.0f} h auto-shutdown", deadline)):
    print(f"    {name:55s} ~{h:4.1f} h each -> ${h * (p_arm + p_x86 + ebs_per_h + ipv4_per_h):.2f}")
EOF

log "dry run of cloud/provision.sh"
"$CLOUD_DIR/provision.sh" --dry-run
