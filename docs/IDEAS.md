# 아이디어 백로그

당장 구현하지 않지만, 앞으로 설계에 영향을 줄 수 있어 기록해두는 아이디어 목록.

## 스타일 모방 / 대련용 정책

기본 목표(강한 정책 하나 학습)와는 별개로 관심 있는 방향:

- **사용자 스타일 모방 정책**: 사용자 본인의 기보를 학습해서 사용자의 대국 스타일 — 습관적인 약점 포함 — 을 모방하는 정책. self-play만으로는 안 되고 사용자 기보에 대한 지도학습(또는 fine-tuning)이 필요할 것으로 보임.
- **약점 노출용 대련 정책**: 사용자를 상대로 이기는 것보다, 사용자가 자주 틀리는 패턴을 잘 드러내도록 최적화된 스파링 정책. "스타일 모방 정책"을 사용자의 근사 모델로 삼고, 그걸 상대로 한 적대적 학습으로 만들 수 있을 듯.

## 사람과의 대국을 통한 학습 가속 실험

- **사람 vs 순수 self-play 학습 속도 비교**: 사람이 직접 두면서(가르치면서) 학습시켰을 때와, 규칙만 주고 AI 혼자 self-play로 학습시켰을 때, 특정 실력 수준에 도달하는 데 걸리는 시간을 비교. "사람과의 대국"도 rollout에서 한쪽 side가 사람인 경우이므로, 위 "서로 다른 두 정책" 인터페이스에서 사람 입력도 하나의 policy처럼 다룰 수 있어야 함.
- **설명 가능한 semantic을 가진 모델**: 사람에게 자신의 판단 근거(왜 이 수를 뒀는지)를 설명할 수 있는 모델. 순수 성능 지표만으로는 안 잡히는 목표라, network 구조나 학습 목표에 interpretability 관련 요소를 넣어야 할 수도 있음 (아직 구체적 방법 미정).
- **실력 측정 문제**: 위 비교를 하려면 실력을 객관적으로 측정할 방법이 필요.
  - 절대적 측정: 고정된 기준(예: 특정 depth의 Stockfish 등 레퍼런스 엔진)과 대국시켜 승률/Elo 추정.
  - 상대적 측정: 서로 다른 시점의 체크포인트끼리(혹은 사람-학습 버전 vs self-play-학습 버전) 라운드로빈 대국시켜 Elo 계산.
  - 어느 쪽이든 rollout 모듈의 "두 정책 롤아웃" 인터페이스 위에서 구현 가능 — arena 평가 로직과 사실상 같은 기능.
  - **TODO(우선순위 낮음→점점 높아질 것)**: 지금(2026-07-12 시점, MCTS 추론 적용 직후)은 사람이 직접 두면서 개선 여부를 체감으로 판단해도 충분할 만큼 명백히 나쁜 지점(REINFORCE 5-step, replay buffer 없음 등)을 고치는 단계라 표준화된 평가가 급하지 않음. 하지만 replay buffer, MCTS 학습 target 통합 등으로 개선분이 점점 미묘해지면 체감 평가로는 한계가 오므로, 그 시점 전에 위 절대적/상대적 측정 중 하나를 arena 평가 모듈로 구현해둘 것.

## 구현됨: 판마다 학습하는 재미용 AI (OnlineValuePolicy)

- 사람과 대국하면서 매판 결과로 policy+value head를 함께 온라인으로 업데이트하는 AI를 만들어 재미 삼아 시각화(`chess_rl/rollout/online_value_policy.py`, viz `/play` 화면).
- 명시적으로 재현성 인프라(ExperimentConfig, seed 고정, run 디렉토리)를 쓰지 않기로 함 — 사람과의 실시간 대국마다 다르게 학습되는 게 목적이라 재현 불가능함을 감수.
- value head: Monte-Carlo 회귀(그 국면 차례 관점의 최종 결과). policy head: 게임에 나온 **모든 수(사람이 둔 수 포함)**를 그 판의 결과로 가중해 강화(REINFORCE with value-baseline 변형) — 순수 on-policy REINFORCE가 아니라 "이긴 판의 수는 흉내내고 진 판의 수는 피하는" 결과-가중 모방 학습에 가까움. 처음엔 "AI 자신이 둔 수만" 학습 대상으로 했다가, 사람과의 대국으로 학습을 가속한다는 위 아이디어와 맞닿아 있어서 사람 수도 포함하도록 바꿈.
- viz에 이미 붙어있는 시각화: 화살표(AI가 직전에 검토한 후보 수, 가치 높을수록 진하게 — policy가 실제로 고르는 수와는 별개), value 게이지, 판 내 value 추이 차트, 판별 학습 loss(전/후) 비교, 누적 학습 판 수 카운터.

## OnlineValuePolicy 학습 가속 아이디어 (MCTS + replay buffer + 비동기 학습)

"판 한 판 둬도 실력이 느는 느낌이 안 든다"는 문제의식에서 나온, 서로 결합하면 시너지가 날 것으로 보이는 아이디어 묶음.

- **추론은 얕은/없는 MCTS, 학습은 깊은 MCTS**: 사람과 실시간 대국할 때는 지금처럼 MCTS 없이(혹은 아주 얕게) 반응성을 유지하고, 학습용 target을 만들 때만 깊은 MCTS를 돌려서 "방문 횟수 분포" 같은 조밀하고 저분산인 policy target을 얻자는 아이디어. 지금 policy head는 게임 결과 하나로만 advantage를 매기는 REINFORCE라 신호가 희소·고분산인 게 학습이 느린 주요 원인으로 추정됨(대화 중 논의, 코드 확인은 아님 — `chess_rl/rollout/online_value_policy.py`의 `learn_from_game` 참고).
  - MCTS 모듈 자체가 아직 없음(`[ ] mcts`, CLAUDE.md 진행 상황). 실제 설계는 별도로 먼저 합의 필요.
  - 재분석 대상을 사람과의 실제 대국으로 한정할지, 별도 self-play 대국도 백그라운드에서 생성해 데이터를 늘릴지는 미정.
- **replay buffer**: 지금은 판 하나 끝나면 그 판 데이터만으로 학습하고 버림(`learn_from_game`이 매번 그 판의 moves만 받음). 여러 판의 포지션을 버퍼에 모아뒀다가 섞어서 학습하면 표본이 늘고 최근 판 하나의 노이즈에 덜 휘둘릴 것으로 기대.
- **여러 판을 모아 배치 학습**: replay buffer와 맞물려서, 판 하나 끝날 때마다 학습하는 대신 N판 모아서 한 번에 배치로 학습하는 방식도 고려.
- **비동기(백그라운드) 학습**: 사람과 다음 판을 두는 동안 백그라운드 스레드/프로세스에서 (MCTS 재분석 + replay buffer 학습을) 동시에 돌리는 방식. 위 세 아이디어가 결합되면: 사람이 계속 대국하는 동안, 쌓인 replay buffer를 깊은 MCTS로 재분석해 배치로 계속 학습 → 대국 텀 없이 계속 실력이 느는 구조 기대.
  - > **[해석/미검증]** RTX 4060 한 장에서 추론(사람과의 실시간 대국, 지연 민감)과 학습(배치, 처리량 중심)을 동시에 돌려도 괜찮을지는 현재 모델 크기가 작아서(`channels=64, num_blocks=4`) 이론상 여유가 있을 것으로 보이지만, 실제 GPU 메모리·CUDA context 공유·Python GIL 하에서의 스레드 경합은 프로파일링 전에는 확인되지 않은 가정임. 실제 구현 전에 벤치마크로 검증 필요.
  - 처음부터 진짜 비동기(스레드/프로세스 분리)로 시도해볼지, 먼저 "판 종료 시 동기적으로 재분석 후 학습"으로 단순하게 시작해 병목을 확인한 뒤 비동기로 옮길지는 선택 가능 — 사용자는 처음부터 비동기를 시도해보고 싶어함.
- **백그라운드 self-play 병행**: 사람과의 실제 대국 재분석뿐 아니라, 사람이 대국하는 동안 백그라운드에서 현재 정책끼리(또는 현재 정책의 서로 다른 체크포인트끼리) self-play 대국도 계속 생성해서 replay buffer에 함께 쌓는 아이디어. 사람과의 대국은 판 수가 적고 사람 개입으로 노이즈가 커서 데이터가 부족한데, self-play는 사람 개입 없이 계속 돌릴 수 있어 데이터량을 늘리는 데 도움이 될 것으로 기대. 위 "두 정책 롤아웃" 인터페이스(`rollout/`)를 그대로 재사용 가능 — self-play는 이미 "양쪽에 같은 정책을 꽂은 특수 케이스"로 다루기로 한 설계와 맞음.

### MCTS 최소 버전 설계 (2026-07-12 논의, 사용자 검토 예정)

자료구조/탐색 알고리즘만 우선 합의, 성능 최적화·학습 target 통합은 별도 단계로 미룸.

- 노드: 명시적 트리 객체, `children: {move: Node}` + edge별 `N`(방문 횟수)/`W`(누적 value)/`Q=W/N`/`P`(prior). transposition table 없음(단순 트리, AlphaZero 원조 방식).
- Selection: PUCT `argmax(Q + c_puct * P * sqrt(N_parent) / (1+N))`.
- Expansion+Evaluation: 리프에서 `encode_board`/`legal_move_mask`로 network 1회 호출 → 합법수만 남긴 policy softmax를 prior로, value 추출. 종료 국면은 network 대신 실제 결과 사용.
- Backup: 리프→루트 경로 따라 `N += 1`, `W += value`, 매 ply 부호 반전.
- 수 선택: 이번 최소 버전은 방문 횟수 argmax(greedy)만. temperature/Dirichlet noise는 self-play 데이터 생성 붙일 때 추가.
- 학습 target 통합(`OnlineValuePolicy`와 연결)은 범위 밖 — 이번엔 `run(board, num_simulations) -> (방문분포, root value)` 인터페이스까지만.

**TODO — 성능 프로파일링(나중에)**: 실제 구현 후 다음을 프로파일링할 것.
- `python-chess` board push/pop이 순수 파이썬이라 트리 노드마다 board 상태 관리 비용이 병목이 될 수 있음.
- 리프 평가를 시뮬레이션마다 1개씩 network에 넣는 게 아니라, 여러 leaf를 모아 배치로 GPU에 넣는 게(virtual loss 필요) 유의미하게 빠른지.
- 실시간 대국(얕은 탐색)과 학습용 재분석/self-play(깊은 탐색)에서 시뮬레이션 수를 얼마나 다르게 가져가야 응답성과 학습 속도를 둘 다 만족하는지.

**진행 상황(2026-07-12)**: 위 배치 leaf 평가는 `mcts.search.run_batched()`/`eval.arena.play_match()`로 구현 완료(순차 대비 20게임·50sim 기준 약 2.7배). 그 후 cProfile로 40게임·100sim(server 모델 크기, GPU) 실측한 결과:
- **network 연산(conv2d+batchnorm+linear) 자체는 전체 시간의 6%뿐.** 배치화는 의도대로 동작.
- `.item()`을 board/legal move마다 개별 호출해 GPU-CPU 동기화를 900만 번 넘게 유발하던 게 전체의 26%(51.9초)로 1위 병목 — 배치 전체를 한 번에 `.cpu().numpy()`로 내리는 방식으로 수정, 마이크로벤치마크로 `_evaluate_batch` 자체는 약 2.9배 빨라짐 확인(commit 참고).
- 그런데도 100게임 전체 시간은 크게 안 줄었는데(노이즈 있는 end-to-end 측정이라 12% 정도), **남은 병목이 python-chess 쪽으로 이동**했기 때문 — legal move 생성(`generate_pseudo_legal_moves`/`_is_safe` 등), `encode_board`의 `piece_map()`, MCTS selection의 `_puct_score`/`max()`/`Node` 할당 등 순수 Python 오버헤드가 지배적. 다음 최적화 대상은 이쪽(예: 국면당 legal_moves 중복 계산 제거, `Node`에 `__slots__` 적용) — 아직 미착수.

## value-delta 가중 policy 학습 (비표준 기법, 2026-07-12 논의·채택)

### 배경: 고정 배치 다스텝 REINFORCE의 발산 (실측으로 확인된 사실)

train_epochs를 크게 올리는 실험(20 → 25000) 과정에서, 13-plane 인코딩 도입 후 격리 실험으로 다음을 확인:
- value 단독 학습(MSE만): 완벽 수렴 (eval MSE 0.0006, 폴스메이트 4포지션 배치).
- value + REINFORCE policy 동시 학습: value가 전부 +1로 포화되어 죽음 (eval MSE ≈ 2.0). gradient clipping(1.0)으로도 안 고쳐지고, policy loss 가중 0.1로도 부분 개선뿐.
- 재현 스크립트: `.tmp_scripts/debug_value_fit.py`.

> **[해석]** 원인: REINFORCE loss `-log π(a|s) × advantage`는 advantage가 음수인 샘플에서 log π → -∞로 보낼수록 loss가 무한히 감소하는, 아래로 비유계 구조. REINFORCE의 이론적 전제는 "배치당 1 gradient step"(on-policy)인데 같은 배치로 수천 스텝을 돌리니 optimizer가 이 내리막을 폭주 — 커진 gradient가 공유 trunk의 활성값을 키우고 value head의 Tanh를 포화시켜(gradient 소실) value 학습까지 파괴. 12-plane 시절부터 있었지만 value가 어차피 못 맞추던 시절이라(차례 정보 부재) 드러나지 않았던 것.

### 검토한 대안들

- **policy 1스텝 + value 다스텝**: REINFORCE 전제 준수. 단, "value를 배치에 수렴시킨 *뒤* policy 스텝"은 advantage ≈ 0이 되어 무의미 — 순서는 "policy 먼저(직전 판까지 수렴된 value를 baseline으로), value 수렴은 나중"이어야 함.
- **PPO-clip**: 표준 해답. old policy 대비 확률 변화율을 샘플당 1±ε로 직접 제한 — 다스텝에서도 폭주 불가. 검증 충분, 구현은 old log prob 저장 + clip 몇 줄.
- **TD advantage**: advantage를 최종 결과 대신 V(s_{t+1}) − V(s_t)로 — 신호가 조밀·저분산해짐. 단 발산 문제 자체의 해결책은 아니라 위 기법과 조합 필요.
- **AWR**: policy 가중치를 exp(advantage/β) ≥ 0으로 — loss 유계화. 대신 "나쁜 수 밀어내기" 신호 소실.

### 채택: value-delta 가중 (사용자 제안)

매 epoch마다 policy 가중치를 "이번 epoch에 value 예측이 움직인 양"으로 쓴다:

```
delta_t(s) = pred_t(s) − pred_{t−1}(s)   (detach)
policy_loss_t = −(log π(a|s) × delta_t).mean()
```

- **advantage와의 관계**: value가 target y로 수렴하면 epoch별 delta의 총합이 telescoping으로 `V_final − V_initial ≈ y − V_before = advantage`. 즉 총 policy 업데이트량이 advantage로 유계 — 기존 REINFORCE의 "같은 신호 무한 반복"이 구조적으로 불가능.
- **self-annealing**: value가 수렴할수록 delta → 0이라 policy 업데이트도 자연 감쇠. value가 포화로 망가지면 delta = 0이 되어 policy가 밀리지 않는 음의 되먹임 — 기존의 "value 망가짐 → advantage 계속 큼 → policy 더 폭주" 악순환이 끊김.
- **구현 단순성**: 직전 epoch의 pred만 기억하면 됨. PPO처럼 old policy log prob 저장/ratio 계산 불필요. 첫 epoch은 delta가 없으므로 value-only.

> **[해석/위험 — 비표준 기법임]** 검증된 레퍼런스 없음. 알려진 우려: (1) value가 배치 내 다른 샘플 때문에 움직이면(일반화 효과) 개별 샘플 가중치 부호가 일시적으로 엉뚱할 수 있음, (2) value가 진동하면 |delta| 합이 총 변화량을 초과해 유계성이 약해짐, (3) PPO는 샘플당 확률 변화율을 직접 제한하는 반면 이건 총량을 간접 제한하는 것이라 보장이 한 단계 느슨함. 실험으로 확인하고, 안 되면 PPO-clip으로 전환하기로 함.

## 수 선택 전략 비교 실험: visits vs q_among_visited (2026-07-12, 결과 유의하지 않음)

시뮬레이션이 적을 때(50~200회) 방문 횟수 argmax보다 "충분히 방문된 수들 중 Q argmax"
(`q_among_visited`, 최다 방문의 25% 이상 방문된 수만 후보)가 나을 것이라는 가설을 arena로 실측.

- **조건**: 같은 checkpoint(human_online_20260712T195638의 5판째, 13-plane + 재료 블렌드 모델), 100판 x 3회, mcts_simulations=50, 색 50/50 교대, 두 정책 모두 확률적 선택(다양성용). 재현 스크립트 `.tmp_scripts/selector_arena.py`.
- **결과**: `q_among_visited` 50승 / `visits` 43승 / 207무 (총 300판).
- **결론: 유의하지 않음.** 승부 난 93판 중 50승은 반반 가정에서 벗어나지 않는 범위이고, trial별로도 17-14, 22-12, 11-17로 방향이 뒤집힘. 기본 선택 전략은 `visits` 유지.
- 두 전략은 `mcts.search.MOVE_SELECTORS`에 병치돼 있어 이름만 바꿔 교체/재실험 가능(`OnlineValuePolicy(move_selector=...)`, `arena.play_match(selector_a=..., selector_b=...)`).

> **[해석]** 지금은 양쪽 모델이 몇 판밖에 학습 안 된 상태라 Q 추정치 자체의 질이 낮아 전략 차이가 드러나기 어려운 조건일 수 있음. 모델이 충분히 학습된 checkpoint가 쌓이면(수십 판 이상) 재실험 가치 있음. 이 문제를 구조적으로 다룬 표준 기법은 Gumbel MuZero(적은 시뮬레이션 예산용 탐색/선택 개선) — 큰 작업이라 당장은 보류.

## 설계에 대한 함의

- 표준 AlphaZero self-play는 **정책 하나**가 자신과 대국(양쪽 다 같은 network)하는 구조. 하지만 위 아이디어들은 **서로 다른 두 정책끼리 롤아웃**하는 걸 요구함(예: 스타일 모방 정책 vs 대련 정책, 혹은 스타일 모방 정책 vs 현재 최강 정책).
- 그래서 이 대국 생성 모듈은 이름을 `selfplay`가 아니라 `rollout`으로 정함(`chess_rl/rollout/`). "양쪽 다 동일 정책"을 하드코딩하지 않고, 처음부터 **양 side에 서로 다른 정책을 꽂을 수 있는 인터페이스**로 설계 — 표준 self-play는 이 인터페이스에서 "양쪽에 같은 정책을 꽂은 특수 케이스"가 되도록.
- arena 평가(신규 vs 기존 최고 정책 대국), 사람과의 대국, Elo 측정용 라운드로빈 대국 모두 본질적으로 "두 정책(또는 정책+사람) 롤아웃"이라, 위 인터페이스가 있으면 전부 같은 코드 경로를 재사용할 수 있을 것.
- 실력 측정(Elo 등)은 rollout보다는 별도 평가 모듈/유틸리티가 될 가능성이 높음 — 실제 설계 시점에 어디 둘지 결정 필요.
