# Simulation-Based RL for Liquidity-Constrained Lending Allocation

Final topic:

```text
Simulation-Based Reinforcement Learning for Bank-Level
Liquidity-Constrained Monthly Lending Allocation
```

The project no longer treats Lending Club history as logged trajectories for
direct offline-RL evaluation. Historical data is used to calibrate a simulator:

```text
Lending Club Data
-> Default Probability Model
-> Risk x Amount x Term Segments
-> Calibrated Bank Lending Simulator
-> PPO Capital Allocation Policy
-> Monte Carlo Evaluation and Stress Tests
```

The defensible claim is:

```text
The RL policy is evaluated in a simulated lending environment calibrated from
Lending Club data, not proven optimal in the real Lending Club market.
```

## Role Split

```text
ML predicts risk.
RL allocates bank capital.
```

ML2 loan-condition recommendation was removed. The RL policy now makes the
bank-level monthly allocation decision.

## Main MDP

State:

```text
S_t = (B_t, G_t, E_t, theta)
```

- `B_t`: current bank liquidity
- `G_t`: current monthly applicant demand by borrower segment
- `E_t`: outstanding exposure by segment and remaining maturity bucket
- `theta`: segment calibration parameters observed by the actor

Default configuration:

```text
J = 5 risk buckets x 3 amount tiers x 2 term buckets = 30 segments
H = 6 maturity buckets
theta_j = (PD_j, interest_j, recovery_j, expected_profit_rate_j)
state dimension = 1 + J + JH + 4J = 331
```

The final `4J` observation entries expose segment-level simulator parameters
to the PPO policy, reducing the oracle-information gap with calibration-aware
myopic baselines during validation, test, and stress evaluation.

Action:

```text
A_t = x_t in R_+^J
0 <= x_t,j <= G_t,j
```

The default experiment uses a soft liquidity reserve, so the policy may cross
the reserve and pay a penalty. The reserve is not exposed as a CLI mode switch
in the final pipeline.

The PPO actor emits raw values that are converted inside the Gymnasium
environment:

```text
actor(S_t) -> (u_t, w_t) -> feasible allocation x_t
```

`u_t` controls how much deployable liquidity to use. `softmax(w_t)` controls
the segment allocation shares.

Transition:

```text
B_{t+1} = B_t + repayments + recoveries
          - funding/servicing costs - new disbursements
E_{t+1} = UpdateLoanBook(E_t, x_t, defaults, repayments)
```

Reward:

```text
RequiredLiquidity_t = B_min + kappa * OutstandingExposure_t
period_profit = interest income - default loss - funding cost - servicing cost
R_t = period_profit / initial_cash
      - lambda * ([RequiredLiquidity_t - B_{t+1}]_+ / initial_cash)^2
```

Monthly interest income, funding cost, and servicing cost are all computed on
beginning-of-month outstanding exposure. This is equivalent to treating default
events as settling at month end. The simulator reports only `funding_cost` and
`servicing_cost`; the older `operating_cost` alias is not emitted.

At the terminal decision month, the simulator runs the remaining loan book
forward with no new lending and adds discounted runoff profit/loss. Any
residual loan-book exposure after runoff receives a terminal expected-NPV
adjustment. Outputs report `decision_period_profit`,
`terminal_runoff_profit`, and `terminal_value` separately; `cumulative_profit`
is their sum.

Training uses the soft liquidity penalty above. Evaluation additionally checks
a simplified capital adequacy ratio:

```text
capital adequacy ratio = capital / risk_weighted_assets
risk_weighted_assets = outstanding loan-book exposure
minimum capital adequacy ratio = 8%
```

Liquidity shortfall and BIS-style capital adequacy are not the same metric. The
evaluation rule treats both as failure modes, but reports their triggers
separately. If capital adequacy falls below 8%, the episode is terminated by
`capital_adequacy` liquidation. If cash falls below required liquidity first,
the episode is terminated by `liquidity_shortfall` liquidation, modeling a
profitable-but-illiquid failure. Liquidity liquidation applies a 60% haircut to
current assets:

```text
current_assets = cash_budget + outstanding loan-book exposure
liquidity_liquidation_loss = 0.60 * current_assets
liquidation_proceeds = 0.40 * current_assets
```

After liquidation, no terminal runoff or terminal value is credited. The
liquidation loss is subtracted from evaluation cumulative profit.

For clarity, outputs report both `min_liquidity_coverage_ratio`
(`budget / required_liquidity`) and `min_capital_adequacy_ratio`
(`capital / risk_weighted_assets`). A positive liquidity shortfall can coexist
with a healthy capital adequacy ratio. Policy result tables also include
`episode_count`, `liquidated_episode_count`,
`capital_liquidation_episode_count`, and
`liquidity_liquidation_episode_count`, so the trigger and number of liquidated
episodes are explicit.

Borrower risk scores are used to define risk buckets. Segment default
probabilities are then calibrated from empirical segment-level bad-loan rates,
not from `mean(risk_score) * scale`.

Because the full state-action space is continuous and high-dimensional, exact
DP is not the main method. PPO is used as an approximate actor-critic method.

## Key Files

- `train_segment_ppo.py`: main PPO training, evaluation, and stress-test script
- `lending_rl_agent/segment_simulator.py`: segment calibration, Gymnasium environment, baselines, rollout evaluation
- `lending_rl_agent/ppo.py`: lightweight PyTorch PPO implementation
- `lending_rl_agent/data.py`: Lending Club loading, leakage-aware cleaning, temporal risk scoring

Older threshold/DQN experiments have been moved under `legacy/` so the root
workspace only exposes the final segment-level PPO methodology.

## Install

```bash
python3 -m pip install -r requirements-neural.txt
```

`gymnasium` is required for the segment allocation environment.

## Quick Smoke Test

```bash
python3 train_segment_ppo.py \
  --max-loans 20000 \
  --episodes 4 \
  --rollout-episodes 2 \
  --eval-episodes 2 \
  --horizon-months 6 \
  --hidden-size 64 \
  --minibatch-size 64 \
  --ppo-epochs 2 \
  --output-dir outputs/segment_ppo_final_smoke
```

Expected structural check:

```text
state dimension = 331
raw action dimension = 31
allocation dimension = 30
```

## Main Training Run

```bash
python3 train_segment_ppo.py \
  --max-loans 120000 \
  --episodes 300 \
  --rollout-episodes 8 \
  --eval-episodes 100 \
  --horizon-months 36 \
  --hidden-size 256 \
  --minibatch-size 1024 \
  --ppo-epochs 6 \
  --liquidity-penalty-lambda 10 \
  --reserve-exposure-ratio 0.05 \
  --terminal-runoff-months 60 \
  --output-dir outputs/segment_ppo_final
```

Repeated runs reuse scored-loan cache files under `outputs/cache`. Use
`--refresh-cache` after changing data preparation or segment calibration logic.
The final training pipeline always uses reservoir sampling, scaled monthly
demand, soft reserve penalties, terminal runoff with terminal expected value,
and stress-augmented train simulators. The actor therefore sees base and stress
segment parameters during PPO training without requiring separate methodology
switches.

## Lambda Sweep

Run PPO trainings for different liquidity penalties and generate the frontier
figure:

```bash
python3 scripts/run_lambda_sweep.py \
  --episodes 400 \
  --output-root outputs/lambda_sweep_400
```

For a longer final sweep, use `--episodes 800`. Extra training arguments can be
passed after `--`, for example:

```bash
python3 scripts/run_lambda_sweep.py \
  --episodes 400 \
  --output-root outputs/lambda_sweep_400 \
  -- --max-loans 120000 --eval-episodes 100
```

The wrapper trains λ values `{0, 0.1, 1, 10, 100}`, writes one run directory per
λ, and saves `lambda_frontier_validation.png` and `lambda_frontier_test.png`
under the sweep `figures/` directory. It also saves shortfall-based frontier
figures using expected shortfall as the liquidity-risk severity axis. Compare
`cumulative_profit_mean` against `liquidity_breach_mean` and
`expected_shortfall_mean` across runs to build the profit-liquidity frontier.
Soft reserve is the default. The reserve threshold is exposure-adjusted:
`B_min + kappa * outstanding exposure`.

To re-score an existing sweep without retraining, use:

```bash
python3 scripts/evaluate_existing_lambda_sweep.py \
  --input-root outputs/lambda_sweep_400 \
  --output-root outputs/lambda_sweep_400_haircut_liquidation_eval
```

This reloads each saved `segment_ppo.pt` checkpoint and reruns only the
validation/test/stress evaluation under the current liquidation-on-breach rule.

## Evaluation

Each policy is evaluated by Monte Carlo rollouts in validation and test
simulators. Baselines:

- `reject_all`
- `fixed_conservative`
- `fixed_aggressive`
- `greedy_expected_profit`
- `budget_aware_heuristic`
- `one_step_lp_closed_form`
- `ppo_rl`

`one_step_lp_closed_form` is the closed-form solution of the one-period linear
allocation problem: with linear per-dollar profit coefficients and one budget
constraint, the LP reduces to sorting segments by expected profit rate.

Metrics:

- cumulative profit
- terminal budget
- terminal runoff profit
- terminal expected NPV
- liquidity breach probability
- minimum liquidity coverage ratio
- expected shortfall
- max shortfall
- capital adequacy ratio
- capital breach count
- max drawdown
- default loss
- funding cost
- servicing cost
- liquidation rate and liquidation loss
- gross and net cash inflow
- disbursed capital
- final-decision disbursement
- final-to-preterminal disbursement ratio
- final disbursement share
- capital utilization
- mean allocation by risk bucket

Evaluation uses common random numbers: every policy is rolled out over the same
episode seed list, reducing comparison noise between PPO and baselines.

Sampling note: reservoir samples carry a demand scale factor so monthly demand
is expanded back toward full-file Lending Club demand. The risk scorer excludes
historical policy/loan-condition variables `int_rate`, `installment`, `grade`,
and `sub_grade`; those columns are still used for simulator calibration.

Stress scenarios:

- default hazard `2x`
- high-risk demand `1.5x`
- starting liquidity `0.7x`
- recovery `0.5x`
- combined stress

## Outputs

The selected output directory contains:

- `metrics.json`: full methodology, calibration, PPO config, validation/test/stress results
- `ppo_training_history.csv`: episode-level PPO training metrics
- `validation_policy_results.csv`: validation policy comparison, including decision/runoff/terminal profit and funding/servicing cost summaries
- `test_policy_results.csv`: holdout scenario policy comparison, including decision/runoff/terminal profit and funding/servicing cost summaries
- `validation_policy_episodes.csv`: rollout-level validation outcomes
- `test_policy_episodes.csv`: rollout-level test outcomes
- `stress_*_policy_results.csv`: stress-test comparisons
- `segment_ppo.pt`: trained PPO checkpoint

## Presentation Claim

Use this wording:

```text
Historical Lending Club data is used to calibrate the simulator, not to
directly evaluate the learned policy's counterfactual value.
```

And:

```text
This is an MDP, so Bellman equations exist. However, exact dynamic programming
is infeasible because the state includes continuous liquidity and
segment-by-maturity loan-book exposure plus segment calibration parameters,
while the action is a 30-dimensional continuous constrained allocation vector.
We therefore use simulator rollouts and train an approximate actor-critic
policy with PPO.
```
