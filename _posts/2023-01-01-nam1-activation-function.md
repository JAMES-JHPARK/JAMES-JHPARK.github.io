---
layout: post
title: "활성화 함수: Sigmoid vs. ReLU vs. Leaky-ReLU"

writer: JHPARK
tags:
  - [활성화함수, Activation Function, Sigmoid, ReLU, Leaky-ReLU]
  
permalink: '/:categories/:title'

toc: true
toc_sticky: true

date: 2023-01-01
last_modified_at: 2023-01-01
---

# 활성화 함수: Sigmoid vs. ReLU vs. Leaky-ReLU

## 1. Sigmoid 대신 ReLU를 사용하는 이유

Sigmoid 함수는 딥러닝에서 오랫동안 사용되어 온 활성화 함수입니다. 하지만 최근에는 ReLU 함수가 Sigmoid 함수보다 더 많은 장점이 있어 널리 사용되고 있습니다.

Sigmoid 함수는 다음과 같이 정의됩니다.

```math
f(x) = \frac{1}{1 + e^{-x}}
```

Sigmoid 함수는 출력이 0과 1 사이의 값을 가지므로, 분류 문제에 적합합니다. 하지만 다음과 같은 단점이 있습니다.

* **포화 문제:** Sigmoid 함수의 출력이 0 또는 1에 가까워지면 기울기가 작아져 가중치와 편향의 업데이트가 어려워집니다.
* **정보 손실:** Sigmoid 함수는 입력 신호의 정보를 일부 손실합니다.

ReLU 함수는 다음과 같이 정의됩니다.

```math
f(x) = max(0, x)
```

ReLU 함수는 Sigmoid 함수의 단점을 해결한 다음과 같은 장점이 있습니다.

* **포화 문제 해결:** ReLU 함수는 입력이 0보다 작으면 출력이 0이 되지만, 입력이 0보다 크면 기울기가 항상 1이므로 포화 문제가 발생하지 않습니다.
* **정보 손실 최소화:** ReLU 함수는 입력 신호의 정보를 최대한 유지합니다.
* **계산 효율성 향상:** ReLU 함수는 Sigmoid 함수보다 계산이 간단합니다.

### 1.1. Sigmoid와 ReLU의 장단점
Sigmoid 함수와 ReLU 함수는 모두 딥러닝에서 사용되는 활성화 함수입니다. Sigmoid 함수는 출력이 0과 1 사이의 값을 가지므로 분류 문제에 적합하지만, 포화 문제와 정보 손실의 단점이 있습니다. ReLU 함수는 출력이 0과 무한대 사이의 값을 가지므로 분류 및 회귀 문제에 모두 적합하지만, 죽은 뉴런의 문제점이 있습니다.

| 특징 | Sigmoid 함수 | ReLU 함수 |
|---|---|---|
| 출력 범위 | 0 ~ 1 | 0 ~ ∞ |
| 적합한 문제 | 분류 | 분류, 회귀 |
| 포화 문제 | 발생 | 발생하지 않음 |
| 정보 손실 | 발생 | 발생하지 않음 |
| 죽은 뉴런 | 발생하지 않음 | 발생할 수 있음 |
| 계산 복잡도 | 높음 | 낮음 |

## 2. 미분불가능한 ReLU의 Back-propagation

ReLU 함수는 x가 0보다 작으면 기울기가 0이 됩니다. 따라서 ReLU 함수는 0에서 미분 불가능합니다.

하지만 역전파(Back-Propagation, BP)는 미분 가능한 함수에만 적용할 수 있는 알고리즘입니다. 따라서 ReLU 함수를 사용하는 신경망에서 BP를 적용하려면 다음과 같은 방법을 사용할 수 있습니다.

* **Heuristic:** ReLU 함수의 미분값을 임의로 정의하는 방법입니다.
* **Gradient clipping:** ReLU 함수의 출력을 0과 1 사이로 제한하는 방법입니다.
* **Leaky ReLU:** ReLU 함수의 기울기를 0보다 작은 값으로 설정하는 방법입니다.

## 3. Leaky-ReLU

Leaky ReLU는 ReLU 함수의 단점을 해결하기 위해 제안된 활성화 함수입니다. Leaky ReLU 함수는 다음과 같이 정의됩니다.

```math
f(x) = max(\alpha * x, x)
```

여기서 alpha는 0보다 작은 값입니다. alpha 값이 작을수록 ReLU 함수와 유사해지고, alpha 값이 클수록 sigmoid 함수와 유사해집니다.

Leaky ReLU 함수는 다음과 같은 장점이 있습니다.

* **포화 문제 해결:** Leaky ReLU 함수는 ReLU 함수와 달리 x가 0보다 작더라도 기울기가 0보다 작은 값을 가지므로, 포화 문제가 발생하지 않습니다.
* **정보 손실 최소화:** Leaky ReLU 함수는 ReLU 함수와 유사하게 입력 신호의 정보를 최대한 유지합니다.
* **계산 효율성 향상:** Leaky ReLU 함수는 ReLU 함수와 유사하게 계산이 간단합니다.

### 3.1. Leaky-ReLU의 alpha값
Leaky ReLU 함수의 alpha 값은 일반적으로 0.01에서 0.1 사이의 값으로 설정합니다. alpha 값이 작을수록 ReLU 함수와 유사해지고, alpha 값이 클수록 sigmoid 함수와 유사해집니다.

alpha 값을 설정할 때는 다음과 같은 사항을 고려해야 합니다.

* **문제의 종류:** 분류 문제에서는 alpha 값을 작게 설정하는 것이 좋습니다.
* **데이터의 분포:** 데이터의 분포가 균등하지 않은 경우에는 alpha 값을 크게 설정하는 것이 좋습니다.
* **학습 과정:** 학습 과정이 안정적으로 진행되지 않는 경우에는 alpha 값을 조정해 보는 것이 좋습니다.

일반적으로는 alpha 값을 0.01에서 0.1 사이의 값으로 설정하고, 필요에 따라 조정해 보는 것이 좋습니다.


