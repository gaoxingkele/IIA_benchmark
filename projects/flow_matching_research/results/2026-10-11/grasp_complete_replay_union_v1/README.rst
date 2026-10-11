GRASP all13 original full-model checkpoint replay union
======================================================

The v4 native worker finished successfully(exit0,no resource abort). Its six
remaining models and24 full scoring recipes join the seven previously passed
full-model replays,covering all13 originally registered trained models/31recipes.
Every recipe retains all3935 validation points and72000 SWAN test points,the
original stride-one windows,batch256,prior/flow settings and saved model/scaler.

An independent NumPy/scikit-learn audit verifies every original/replayed file
hash,all full labels/row identities,finite scores,and their agreement under
originally frozen tolerances. It recalculates AP,AUROC,test-label-oracle BestF1,
validation99 threshold,precision,recall,and no-PA F1 from whole arrays. All original
training results,checkpoints,histories and score artifact hashes also pass.

Independent verification reads full native inference outputs;it does not replace
the actual native model replays. Scores are not universally bitwise identical
across CPU/GPU;the preregistered numerical/prediction tolerances remain unchanged.
Oracle scores and validation-only control metrics remain separate protocols.
No inference repeat is counted as an extra independent training seed.

The separately trained tau0.25 and Transformer-seed1104 native model audits remain
separate evidence. This union proves the original13-model replay obligation only.
It does not certify original paper equivalence,strict TAB performance,all GRASP
data/tasks,or the full45-paper/681-method-baseline-ablation/273-task project.

independent_complete_full_array_verification.json records every recipe and metric.
The original native v4 output remains in ../grasp_full_checkpoint_replay_v4.
