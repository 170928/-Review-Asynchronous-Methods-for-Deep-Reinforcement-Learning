# -Review-Asynchronous-Methods-for-Deep-Reinforcement-Learning
>DeepMind, MILA
>논문 리뷰 및 구현 코드 와 코드별 해석 정리

## [Abstract]
개념적으로 단순하고 가벼운 프레임 워크를 제안하는 논문입니다.  
deep neural network controller의 optimization을 위한 asynchronous gradient descent 를 사용합니다.  
이 asynchronous gradient descent는 4개의 standatrd reinforcement learning algorithm의 변화된 형태입니다.  
parallel actor-learners는 training process를 안정화 시킵니다.  
그리고, 4가지의 standard algorithm들이 neural network controller를 성공적으로 학습되도록 만들어 줍니다.  
actor-critic의 변화된 asychronous variant는 최고의 성능을 보이며, 현재 Atari domain에서 뛰어난 성능 향상을 보여줍니다.  

## [Introduction]
Deep neural network를 통한 reinforcement algorithm은 효과적으로 작동합니다.  
그러나, online RL algorithms과 deep neural network의 조합 (ex, DQN)은 학습에서 다소 불안정한 모습을 보여줍니다.  
이 문제의 원인은 크게 2가지로 볼 수 있습니다.  
(1) non-stationary problem when poilicy updates  
(2) correlation of observed data  
이를 해결하기 위해서, agent의 data를 "experience replay memory"에 저장합니다.  
그리고 추후에 random sampling을 통해서 "replay memory"에서 꺼내 온 batch 를 사용해서 학습을 하는 것으로  
data 간의 correlation을 줄임과 동시에 non-stationary 문제를 해결합니다.  
그러나, 이때 reinforcement algorithm이 "off-policy" 형태로 강제된다는 한계점이 존재합니다.  

이 논문에서는 이 "replay memory"를 대체하여 새로운 패러다임을 제안합니다.  
multiple agents들을 parallel 하게 독립된 multiple instance environment에서 실행합니다.  
이 parallelism은 agent들의 데이터를 decorrelate 시킵니다.  
각각의 agent 들이 주어진 time-step에서 다른 state에 존재하며 다른 경험을 하기 때문입니다.  
이 간단한 idea는 "on-policy" 뿐만 아니라 "off-policy" RL algorithm이 더 많은 영역에서 작동할 수 있도록 해 줍니다.  

asynchronous advantage actor-critic (A3C)는 제안한 method 들 중 가장 뛰어난 성능을 보이며 continous motor 와 같은 continuous action space 환경 뿐 아니라 deterministic action space 환경에서도 모두 뛰어난 성능을 보여줍니다.  

## [Related Work]
General Reinforcement Learning Architecture (Gorila : Nair et al., 2015) 는 distributed setting에서 강화학습 agent들의 asynchronous training 을 수행합니다.  
Gorila에서, 각각의 agent들은 각자의 environment를 가지고, 독립적인 replay memory를 가집니다. 뿐만 아니라 이 memory로 부터 random sampling한 데이터를 활용하여 DQN loss를 사용해서 poilicy를 학습 합니다.  
이렇게 독립적으로 계산된 DQN loss의 DQN parameter로 미분된 gradient들은 "central parameter server"에 전달됩니다.  
server는 model의 central copy를 update 합니다.  
update된 policy parameter들은 fixed interval을 가지고 actor-learners (agents들) 에게 전달됩니다.  

> 정확한 방법은 해당 논문을 참조하는 것이 좋을 것 같습니다. 추후에 포스팅 할 생각입니다.

Li & Schuurmans (2011) 의 연구에서는 Map Reduce framework 를 linear function approximation을 통한 batch reinforcement learning algorithm을 parallel하게 학습합니다.  
Praralleism은 large matrix operation을 빠르게 하기 위해서 사용되지만, agent의 experience의 수집을 parallel하게 만들지는 않습니다.  

Grounds & Kudenko (2008) 은 Sarsa algorithm의 parallel version을 제안했습니다.  
이 알고리즘은 multiple actor-learners를 활용하여 training 과정을 가속화 시킵니다.  
각각의 actor는 독립적으로 학습을 수행하고 주기적으로 학습된 weight를 전달합니다.  
이때, peer-to-peer communication을 사용하여 학습된 weight를 전달하며, 중요한 학습이 이루어진 가중치가 퍼지게 됩니다.  

Tsitsiklis (1994) 는 Q-learning의 asychronous 최적화 환경에서 convergence 특징에 대해서 연구를 하였습니다.  
만약 오래된 (outdated) 데이터가 버려지고 몇가지 추가적인 기술적인 가정이 만족된다면 Q-learning이 여전히 convergence가 보장된다는 것을 보였습니다.  

## [Reinforcement Learning Background]
이 논문에서는 일반적인 강화학습 환경을 고려합니다.  
(1) agent 는 환경 ![image](https://user-images.githubusercontent.com/40893452/45093490-7c437e00-b153-11e8-8149-1e7cb7c81b9c.png) 와 상호작용 합니다.  
(2) discrete time step t 에서 agent는 상호작용합니다.  
(3) agent는 t time step일 때, s(t) 의 state를 받고 가지고 있는 policy 에 따라서 action a(t)를 선택합니다.  
(4) environment는 a(t)에 따라서 estate s(t+1)을 받고 scalar reward r(t)를 돌려 줍니다.  
위의 과정은 agent가 terminal state에 도달할 때까지 반복합니다.  
![image](https://user-images.githubusercontent.com/40893452/45093595-d04e6280-b153-11e8-8030-2ddf6c7c0286.png)는 total accumulated return with discounted factor입니다.  
그러므로, 강화학습의 목표는 각각의 state s(t)에서부터 기대되는 리턴 (expected return) 을 최대화 시키는 정책을 학습하는 것입니다.  

action-value ![image](https://user-images.githubusercontent.com/40893452/45093790-8154fd00-b154-11e8-952a-647cd4f58d96.png)

