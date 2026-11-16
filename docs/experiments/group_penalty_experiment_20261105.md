# Group-penalty experiment (registered before execution)

Objective: test one modification to the existing group-direct approach on two additional past periods, without reading HOLDOUT or using leaderboard values. Existing #1314 group-direct already uses absolute loss and SMP/time weights; it is not a new group-loss model.

Evaluation: April and August 2024, separately; fixed groups from the provided sites, site cohort with complete 24-hour records throughout each month. Cohort is defined using coverage only, not generation values. The April and August cohorts may differ. Use identical cohort, actuals, timestamps, and SMP for all variants. Include no HOLDOUT data; allow only already separated files with parquet row-group timestamp maximum before lock.

Group-direct baseline: #1314's three fixed HGB absolute-loss configurations, group capacity factor target, coverage × historical SMP × k sample weights. Variant: same models/features, but replace coverage weight with observed available kW × historical SMP × k. Since capacity differs between groups, this weights each group's historical absolute energy error in the same scale as the provisional grouped penalty. Keep all model settings unchanged; do not search blend weights.

Comparison constraint: actual #1314 uses a team FS2+L1 site forecast. Its out-of-fold predictions outside JUL/OCT/JAN/FEB are not known to be available. A 40/60 blend using an independently fitted site forecast is a **proxy**, not original #1314. Evaluate proxy with old versus new group-direct component plus the standalone group variant. Do not claim a win over real #1314 without matching out-of-sample FS2+L1 predictions. Group-direct training always uses only timestamps through the applicable cutoff; validation observations are never used for fitting.

Metrics: provisional README group penalty and weighted group MAE; lower is better. Report per-month group coverage, configuration, input hashes, fit cutoffs and evidence. April/August are exploratory past months, not a final holdout or a claim about official revenue. The selection of groups/cohort does not establish a feasible 2x20MW portfolio for partial sites.
