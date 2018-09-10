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

action-value ![image](https://user-images.githubusercontent.com/40893452/45093790-8154fd00-b154-11e8-952a-647cd4f58d96.png) 은 state s에서 action a를 선택했을 때 기대되는 리턴 (expected return)을 의미합니다.   
optimal value function 어떤 policy를 따를 때 state s 에서 최대의 return을 주는 action a를 반환 해주는 함수를 의미하며 다음과 같이 표현 됩니다. 
![image](https://user-images.githubusercontent.com/40893452/45093884-c7aa5c00-b154-11e8-895e-f2f06ff41def.png)![image](https://user-images.githubusercontent.com/40893452/45093900-d42eb480-b154-11e8-945d-c4d03ef6a1f5.png)  
유사하게, 임의의 policy 아래에서 state s일 때의 value의 가치는 다음과 같이 표현 됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45093963-fb858180-b154-11e8-8608-aa16ba54296b.png)![image](https://user-images.githubusercontent.com/40893452/45093981-06401680-b155-11e8-9fd9-4135530b3da7.png)  
이는 state s에서 policy를 따랐을떄 따라오게 되는 expected return을 의미합니다.  

value-based model-free reinforcement learning 방법에서, action-value function은 neural network와 같은 function approximator를 활용해서 표현됩니다.   
![image](https://user-images.githubusercontent.com/40893452/45094124-5a4afb00-b155-11e8-8491-9a9b4e698e62.png)는 parameter ![image](https://user-images.githubusercontent.com/40893452/45094143-6931ad80-b155-11e8-9547-39b5683c36d8.png)를 통해서 근사되는 action-value function을 의미합니다.  
![image](https://user-images.githubusercontent.com/40893452/45094143-6931ad80-b155-11e8-9547-39b5683c36d8.png)를 update하는 방법은 다양한 알고리즘들이 존재합니다.  
하나의 예로 DQN 이 존재합니다.  
DQN 에서는 1 step 마다 ![image](https://user-images.githubusercontent.com/40893452/45094143-6931ad80-b155-11e8-9547-39b5683c36d8.png) 로 근사되는 action-value function을 학습합니다.  
이때, neural network로 근사되므로 다음과 같은 loss function을 사용하여 loss를 minimization하는 방향으로 학습이 진행됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45094549-9fbbf800-b156-11e8-9cdb-3242b8b69555.png)  
1 step Q-learning은 action value function을 one step return r(t) + (discounted factor) * max_(a){Q(s(t+1), a)} 을 사용하여 업데이트 합니다.  
이 방법의 단점은 reward r을 얻는것이  s(t)와 a(t)의 state-action pair에 직접적으로만 영향을 미친다는 것입니다.  
Q(s, a)의 업데이트를 통해서 s(t)와 a(t)와는 다른 state와 action pair들은 직접적으로 영향을 받지 않습니다.  
이것은 learning process에서 state와 action을 경험하고 학습하는 과정에서 속도를 느리게 만듭니다.  

이 문제를 해결하기 위한 하나의 방법으로, n-step return (Watkins, 1989, Peng & williams, 1996)에서 제안된 방법이 있습니다.  
n-step Q-learning은 Q(s, a)를 업데이트 하는 과정에서, n개의 앞선 state와 action pair까지 하나의 reward r(t)가 영향을 미치게 합니다.  
그 식은 아래와 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/45094944-c0388200-b157-11e8-8484-89c7b8dee029.png)![image](https://user-images.githubusercontent.com/40893452/45094966-c9c1ea00-b157-11e8-9e67-8bc8b1106743.png)  
이 수식을 통해서, rewrard의 propagation 과정이 더 효율적으로 이루어져서 학습이 빨라지게 됩니다.  

value-based method 와 달리, policy-based model-free mothod는 직접적으로 policy ![image](https://user-images.githubusercontent.com/40893452/45095992-6e452b80-b15a-11e8-8262-f5cc98303374.png)
의 parameter를 학습합니다.   
이때, gradient ascent 방법을 따르며, expected return ![image](https://user-images.githubusercontent.com/40893452/45096150-d72ca380-b15a-11e8-95b2-c8529d81d55c.png)에 대한 gradient를 사용합니다.  

하나의 예로, REINFORCE 알고리즘이 있습니다.  
이 알고리즘은 policy parameter를 업데이트 할 때, ![image](https://user-images.githubusercontent.com/40893452/45096150-d72ca380-b15a-11e8-95b2-c8529d81d55c.png)의 미분 값인 ![image](https://user-images.githubusercontent.com/40893452/45096219-ffb49d80-b15a-11e8-8f74-2fa86edd58d0.png)의 방향으로 학습을 진행합니다.  
> 이때 학습 과정에서 생겨나는 bias는 value function at state s(t)를 R(t) 에서 빼는 것으로 없애게 됩니다. 자세한 내용은 논문을 참조해 주세요.  
즉, 최종적인 학습 과정에서의 gradient는 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/45096326-3b4f6780-b15b-11e8-90a8-dd27329c52a3.png)   

b(t) 를 ![image](https://user-images.githubusercontent.com/40893452/45096396-6f2a8d00-b15b-11e8-8025-8d94d07a1b2c.png)과 같이 근사된 value function으로 대체하는 것을 통해서 "policy gradient"의 variance가 더 낮아지게 됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45096458-91bca600-b15b-11e8-95e4-23c3c173620f.png) 가 baseline (i.e., b(t))로 사용되는 것으로 R(t) - b(t) 의 크기 (quantity)는 policy gradient의 크기를 변화시킵니다.  
그리고 이것은 state s(t)에서의 action a(t)의 advatage의 추정으로써 보여집니다.  
> R(t) 는 Q(s(t), a(t)) 의 추정치이며, b(t) 는 V(s(t))의 추정치 이기 때문에, s(t)의 expected return과 s(t)에서의 a(t)의 expected return 과의 차이를 advantage of action a(t) in state  s(t)라고 표현하는 것 같습니다.  
이 접근법은 actor-critic 구조로써 보여질 수 있습니다.  

## [Asynchronous RL framework]
이 논문에서는 one-step Sarsa, one-step Q-laearning, n-step Q-learning 과 advantage actor-critic의 변형된 알고리즘을 제안합니다.  
이러한 mulit-threaded asynchronous 변형 알고리즘을 제안하는 목적은 policies들의 deep neural network를 통한 안정적인 학습 방법과 larger resource requirement를 줄일 수 있는 방법을 제안하기 위해서 입니다.   
기본적인 RL 알고리즘들과는 다르지만, actor-critic 이 on-policy policy search 알고리즘이 되는것과 Q-learninig이 off-policy value-based method가 되는것과 함꼐 2가지 메인 아이디어를 활용해서 4가지 위에서 언급한 알고리즘들이 실용적이 되도록 합니다.  

1. 이 논문에서는 aasynchronous actor-learner들을 사용합니다. 이는 Gorila framework에서와 유사하지만, 다른 machine과 server를 사용하는 대신에 multiple CPU  thread를 single machine에서 사용합니다.  그리고 이 방법은 gradients와 parameter를 server를 통해 공유하는 cost를 줄여주며, Hogwild! (Recht et al., 2011) 스타일의 업데이트를 training 과정에서 가능하도록 만들어 줍니다.  

2. Multiple actor-learners가 parallel 하게 학습을 수행하는 과정에서 다른 environment를 탐색하게 됩니다.  
그 결과 각각의 actor-learner들은 다른 policy를 가지고 탐색을 수행하게 됩니다.  
다른 thread에서 다른 탐색 정책을 가지고 작동하는 것에 의해서, 전체적인 변화는 online update방식의 parameter 업데이트가 single agent의 online update에 비해서 less correlated 하게 됩니다.  
그러므로, replay memory를 사용하지 않고, parallel actor들의 다른 탐색 정책을 가진 actor 들을 이용하여 데이터의 학습과정에서의 correlation을 줄입니다.  

experience memory가 아닌 multiple parallel actor-learners를 사용하여 학습하는 것에는 몇가지 이점이 있습니다.   
1. parallel actor-learner의 수에서 linear하게 training time이 감소합니다.  
2. experience replay 에 의존하지 않게되어, on-policy reinforcement learning method를 사용할 수 있게 됩니다. (ex) SARSA , actor-critic  

## [Asynchronous one-step Q-laerning]
![image](https://user-images.githubusercontent.com/40893452/45151193-9be8ae00-b208-11e8-9f65-7b9717e7fbcf.png)  
위의 알고리즘은 이 논문에서 제시하는 asynchronous one-step Q-learning 의 수도코드 입니다.  

## [Asynchronous one-step Saras]
asynchronous one-step Sarsa 알고리즘은 asynchronous one-step Q-learning과 Algorithm1과 다른 target value Q(s,a)를 사용한다는 것 빼고 모두 동일합니다.  
one-step Sarsa 에서 사용되는 target value는 r + (discount_factor) * Q( s(t+1) , a(t+1) ; targetNet_paramter ) 입니다.  
target network 가 사용되며 학습을 안정화 시키기 위해서 일정 time step을 주기로 target network를 업데이트 합니다.  

## [Asynchronous n-step Q-learning]
multiple step Q-learning을 다음과 같습니다.   
![image](https://user-images.githubusercontent.com/40893452/45205494-46220d80-b2bd-11e8-8445-76374c9a5830.png)  
이 알고리즘은 forward view ( n step 앞의 결과 )를 본다는 점에서 일반적이지 않습니다.  
이런 forward view를 사용하는 것은 neural network를 학습하는 과정에서 monetum based methods 와 backpropagation 과정에서 훨씬 더 효과적인 학습이 가능하도록 해 줍니다.  
한번의 업데이트를 위해서, 알고리즘은 policy를 기반으로 action을 고리며 최대 t(max) step까지 미리 action을 고릅니다. ( 또는 state가 끝날 때 까지 ).  
이 과정은 agent t(max) 까지의 rewards 를 한번에 마지막으로 update 했던 state에 대한 update 부터 받아옵니다.  

## [Asynchronous advantage actor-critic]
Asynchronous advantage actor-critic (A3C) 알고리즘은 policy 와 value function 모두를 가지고 있습니다.  
위의 논문에서 제시한 것처럼 n-step Q-learning 알고리즘처럼 forward view를 사용해서 policy 와 value function을 업데이트 합니다.  
policy와 value function 들은 모두 t(max) or terminal setate에 도착한 후에 업데이트 됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45300004-3407cf80-b548-11e8-847a-70cfd5fb3e6e.png)  
k 값은 upper-bounded "t(max)" 의 내에서 달라질 수 있다.  

policy의 매개 변수 θ와 value-function의 θv가 분리되어있는 것처럼 보여 지지만 실제로 우리는 실제로 일부 매개 변수를 공유합니다.   
policy는 π (at | st; θ)에 대해 하나의 softmax 출력을 가지는 convolutional neural netowrk를 사용합니다.  
value-function V(st; θv)에 대해 하나의 선형 출력을 가집니다.  
이때, 모든 non-output layer들의 가중치는 공유됩니다.  

정책 π의 엔트로피를 loss function에 더하면 suboptimal 로의 premature (조기,이른) convergence (수렴)을 방지하여 exploration을 개선한다는 것을 발견했다.  
> This technique was originally proposed by (Williams & Peng, 1991)  

policy 매개 변수와 관련하여 "엔트로피 정규화"를 포함하는 완전한 loss function의 gradient가 다음과 같은 형태를 취합니다.  
![image](https://user-images.githubusercontent.com/40893452/45300917-982b9300-b54a-11e8-8422-ad89709e1d88.png)  

## [Optimization]
이 논문에서는 the standard non-centered RMSProp update 를 사용합니다.   
![image](https://user-images.githubusercontent.com/40893452/45300990-d759e400-b54a-11e8-812e-354af1785910.png)  

## [Experiments]

## [Scalability and Data Efficiency]

## [Robustness and Stability]
















