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
