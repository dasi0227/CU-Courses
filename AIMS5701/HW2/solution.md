## Problem 1

### (a)

$$
\begin{align*}
& \because P(X,Y,Z) = P(X,Y)P(Z|X,Y),\,\,P(X,Y)=P(X)P(Y|X) \\
& \therefore P(X,Y,Z) = P(X)P(Y|X)P(Z|X,Y)\\
\\
& \because P(X,Y,Z) = P(X,Y)P(Z|X,Y),\,\,P(X,Y)=P(Y)P(X|Y) \\
& \therefore P(X,Y,Z) = P(Y)P(X|Y)P(Z|X,Y)\\
\\
& \because P(X,Y,Z) = P(Y,Z)P(X|Y,Z),P(Y,Z)=P(Z)P(Y|Z) \\
& \therefore P(X,Y,Z) = P(Z)P(Y|Z)P(X|Y,Z)
\end{align*}
$$

### (b)

$$
\begin{align*}
& \because P(F,W) = P(F)P(W|F)\\[1ex]
& \because P(W) = P(W,F)+P(W,\neg F) = P(W|F)P(F) + P(W|\neg F)P(\neg F)\\[1ex]
& \therefore P(F|W) = \frac{P(F,W)}{P(W)} = \frac{0.002 \times 0.90}{0.90 \times 0.002 + 0.04 \times 0.998} \approx 0.0431
\end{align*}
$$

The probability that the battery is actually faulty given a warning is only about 4.31%. False warnings are approximately 22 times as frequent as genuine fault warnings.
$$
\begin{align*}
\frac{P(W,\neg F)}{P(W,F)} = \frac{P(W|\neg F)\times P(\neg F)}{P(F)P(W|F)} = \frac{0.0399}{0.0018} \approx 22
\end{align*}
$$


## Problem 2

### (a)

$$
\begin{align*}
P(H,C,E,N)=P(H)P(C|H)P(E|C)P(N|C)
\end{align*}
$$

### (b)

$$
\begin{align*}
P(C=1)
&=P(C=1,H=0)+P(C=1,H=1)\\
&=P(H=0)P(C=1|H=0)+P(H=1)P(C=1|H=1)\\
&=0.60 \times0.15 + 0.40 \times 0.85\\
&= 0.43
\end{align*}
$$

$$
\begin{align*}
P(E=1)
&=P(E=1,C=0)+P(E=1,C=1)\\
&=P(C=0)P(E=1|C=0)+P(C=1)P(E=1|C=1)\\
&=(1-0.43) \times 0.25 + 0.43 \times 0.80\\
&= 0.4865
\end{align*}
$$



### (c)

$$
\begin{align*}
P(C=1|E=1)
&=\frac{P(C=1,E=1)}{P(E=1)}=\frac{P(C=1)P(E=1|C=1)}{P(E=1)}\\[1ex]
&=\frac{0.43 \times 0.80}{0.4865} \approx 0.707
\end{align*}
$$



## Problem 3

### (a)

$$
\begin{align*}
H &\to 1 \times (2 - 1) = 1\\
C &\to 2 \times (2 - 1) = 2\\
E &\to 2 \times (2 - 1) = 2\\
N &\to 2 \times (2 - 1) = 2\\
Total &\to 1 + 2 + 2 + 2 = 7
\end{align*}
$$

### (b)

$$
\begin{align*}
E &\to 1 \times (2 - 1) = 1\\
C &\to 2 \times (2 - 1) = 2\\
Total &\to 1 + 2 = 3
\end{align*}
$$

Air-conditioner use increases the likelihood of high electricity consumption for sure, while high electricity consumption may have other causes (such as fridges or lamps instead of the air condition). Therefore, the original causal structure is more intuitive and its conditional probabilities are likely easier for experts to estimate, even though it requires more parameters.



## Problem 4

### (a)

$H \perp\!\!\perp E$ is not guaranteed → The path $H \to C \to E$ is a causal chain. Since C is not observed, the path remains active.

$H \perp\!\!\perp E \,\, | \,\, C$ is guaranteed → The causal chain $H\to C \to E$ is blocked by the observed $C$.

### (b)

$N \perp\!\!\perp H \,\, | \,\, C$ is guaranteed → The causal chain $H\to C \to N$ is blocked by the observed $C$.

$N \perp\!\!\perp E \,\, | \,\, C$ is guaranteed → The fork $N \leftarrow C \to E$  is blocked by the observed $C$.

By the local Markov property, given its parent C, N is conditionally independent of its non-descendants H and E, since they provide no additional information about N once C is known.



### (c)

$X \perp\!\!\perp Y$ is guaranteed → The path $X\to Z\leftarrow Y$ is a collider. Since $Z$ is not observed, the path is blocked.

$X \perp\!\!\perp Y \,\, | \,\, Z$ is not guaranteed → The collider $X\to Z\leftarrow Y$  is open by the observed $Z$.

When high refrigerator temperature is observed, knowing that the door is open may make a faulty thermostat less likely, because the open door provides an alternative explanation for the high temperature.



## Problem 5

### (a)

$$
\begin{align*}
&\because F \perp\!\!\perp O\\
&\therefore P(F=1,O=1)=P(F=1)P(O=1)= 0.0002 \\
&\therefore P(F=1,O=0)=P(F=1)P(O=0)= 0.0098 \\
&\therefore P(F=0,O=1)=P(F=0)P(O=1)= 0.0198 \\
&\therefore P(F=0,O=0)=P(F=0)P(O=0)= 0.9702 \\
&\therefore P(A=1,F=1,O=1)=P(A=1|F=1,O=1)P(F=1,O=1) = 0.000196 \\
&\therefore P(A=1,F=1,O=0)=P(A=1|F=1,O=0)P(F=1,O=0) = 0.00931 \\
&\therefore P(A=1,F=0,O=1)=P(A=1|F=0,O=1)P(F=0,O=1) = 0.01782 \\
&\therefore P(A=1,F=0,O=0)=P(A=1|F=0,O=0)P(F=0,O=0) = 0.004851 \\
&\because P(A=1)=P(A=1,F=1,O=1)+P(A=1,F=1,O=0)\\
&\quad\quad\quad\quad\quad\quad +P(A=1,F=0,O=1)+P(A=1,F=0,O=0) \\
&\therefore P(A=1)= 0.032177
\end{align*}
$$

### (b)

$$
\begin{align*}
P(F=1|U=1,V=1) = \frac{P(F=1,U=1,V=1)}{P(U=1,V=1)} \\[1ex]
\end{align*}
$$

First:
$$
\begin{align*}
&P(U=1,V=1) = P(U=1,V=1,A=1)+P(U=1,V=1,A=0) \\[1ex]
&P(U=1,V=1,A=1) = P(U=1,V=1|A=1)P(A=1) \\[1ex]
&P(U=1,V=1,A=0) = P(U=1,V=1|A=0)P(A=0) \\[1ex]
&U \perp\!\!\perp V \,\, | \,\,A \\[1ex]
&P(U=1,V=1|A=1)=P(U=1|A=1)P(V=1|A=1) \\[1ex]
&P(U=1,V=1|A=0)=P(U=1|A=0)P(V=1|A=0) \\[1ex]
\end{align*}
$$








Second:
$$
\begin{align*}
&P(F=1,U=1,V=1)=P(F=1,U=1,V=1,A=1)+P(F=1,U=1,V=1,A=0) \\[1ex]
&P(F=1,U=1,V=1,A=1)=P(U=1,V=1|F=1,A=1)P(F=1,A=1) \\[1ex]
&P(F=1,U=1,V=1,A=0)=P(U=1,V=1|F=1,A=0)P(F=1,A=0) \\[1ex]
&F \perp\!\!\perp (U,V) \,\, | \,\,A \\[1ex]
&P(U=1,V=1|F=1,A=1)=P(U=1,V=1|A=1) \\[1ex]
&P(U=1,V=1|F=1,A=0)=P(U=1,V=1|A=0) \\[1ex]
&P(F=1,A=1)=P(F=1,A=1,O=1)+P(F=1,A=1,O=0) \\[1ex]
&P(F=1,A=0)=P(F=1)-P(F=1,A=1) \\[1ex]
\end{align*}
$$
Therefore:
$$
\begin{align*}
P(F=1|U=1,V=1) = \frac{0.0074339884}{0.0257431078} \approx 0.288776
\end{align*}
$$



## Problem 6

### (a)

In a Bayesian network, a node is conditionally independent of its non-descendants given its parents. Once the parents are observed, the non-descendants provide no additional information about the node.

In Graph A, \(C\) is the parent of \(N\), while \(H\) and \(E\) are non-descendants of \(N\). Therefore: $H \perp\!\!\perp N \,\, | \,\,C$ and $E \perp\!\!\perp N \,\, | \,\,C$

### (b)

$$
\begin{align*}
X \perp\!\!\perp W \,\, &| \,\,Y \\
Z \perp\!\!\perp W \,\, &| \,\,Y \\
Z \perp\!\!\perp Y \,\, &| \,\,X \\
Z \perp\!\!\perp W \,\, &| \,\,X \\
X \perp\!\!\perp W \,\, &| \,\,Y,Z \\
Z \perp\!\!\perp W \,\, &| \,\,X,Y \\
Z \perp\!\!\perp Y \,\, &| \,\,X,W \\
\end{align*}
$$

