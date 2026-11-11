#!/bin/sh
# usage: validate.sh <candidate_dir>  -> writes validator_report.json and official_validator.log into it
C="$1"; D=/c/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data
(cd /c/dev/icn-solar-oibc && python scripts/oibc_submission_validator.py --sites $D/sites.csv --composition "$C/composition.csv" --forecast "$C/forecast.csv" --evaluation-smp $D/smp.csv --rule scripts/oibc2026_capacity_rule_v0_1.json --expected-time-count 8760 --input-capacity-unit kW > "$C/validator_report.json"); echo "team rc=$? passed=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['passed'])" "$C/validator_report.json")"
(cd /c/Users/dkrak/Documents/Codex/OIBC && python -I -X utf8 official_validator_20261102.py --composition "$C/composition.csv" --forecast "$C/forecast.csv" --data $D > "$C/official_validator.log" 2>&1); echo "official rc=$?"; head -1 "$C/official_validator.log"
