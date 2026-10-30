### Problem 1

#### (a) 

$f_1(H) = P(H)$ → Scope：$\{H\}$

| $H$  | $f_1(H)$ |
| ---- | -------- |
| 0    | 0.6      |
| 1    | 0.4      |

$f_2(H,V)=P(V|H)$ → Scope：$\{H,V\}$

| $H$  | $V$  | $f_2(H,V)$ |
| ---- | ---- | ---------- |
| 0    | 0    | 0.8        |
| 0    | 1    | 0.2        |
| 1    | 0    | 0.3        |
| 1    | 1    | 0.7        |

$f_3(V,F)=P(F|V)$ → Scope：$\{V,F\}$

| $V$  | $F$  | $f_3(V,F)$ |
| ---- | ---- | ---------- |
| 0    | 0    | 0.9        |
| 0    | 1    | 0.1        |
| 1    | 0    | 0.2        |
| 1    | 1    | 0.8        |

$f_4(H,V)=P(A=1|H,V)$ → Scope：$\{H,V\}$

| $H$  | $V$  | $f_4(H,V)$ |
| ---- | ---- | ---------- |
| 0    | 0    | 0.05       |
| 0    | 1    | 0.50       |
| 1    | 0    | 0.60       |
| 1    | 1    | 0.90       |









#### (b)

【Join】$f_{124}(H,V)=f_1(H)f_2(H,V)f_4(H,V)=P(A=1,H,V)$ → Scope：$\{H,V\}$

| $H$  | $V$  | $f_{124}(H,V)$                       |
| ---- | ---- | ------------------------------------ |
| 0    | 0    | $0.8 \times 0.6 \times 0.05 = 0.024$ |
| 0    | 1    | $0.2 \times 0.6 \times 0.50 = 0.060$ |
| 1    | 0    | $0.3 \times 0.4 \times 0.60 = 0.072$ |
| 1    | 1    | $0.7 \times 0.4 \times 0.90 = 0.252$ |

【Eliminate】$f_{5}(V)=\sum_{H}f_{124}(H,V)=P(A=1,V)$ → Scope：$\{V\}$

| $V$  | $f_{5}(V)$              |
| ---- | ----------------------- |
| 0    | $0.024 + 0.072 = 0.096$ |
| 1    | $0.060 + 0.252 = 0.312$ |

【Join】$f_{35}(V,F)=f_3(V,F)f_5(V)=P(A=1,V,F)$ → Scope：$\{V,F\}$

| $V$  | $F$  | $f_{35}(V,F)$               |
| ---- | ---- | --------------------------- |
| 0    | 0    | $0.096 \times 0.9 = 0.0864$ |
| 0    | 1    | $0.096 \times 0.1 = 0.0096$ |
| 1    | 0    | $0.312 \times 0.2 = 0.0624$ |
| 1    | 1    | $0.312 \times 0.8 = 0.2496$ |

【Eliminate】$f_{6}(F)=\sum_{V}f_{35}(V,F)=P(A=1,F)$ → Scope：$\{F\}$

| $F$  | $f_{6}(F)$                 |
| ---- | -------------------------- |
| 0    | $0.0864 + 0.0624 = 0.1488$ |
| 1    | $0.0096 + 0.2496 = 0.2592$ |

#### (c)

$$
\begin{align*}
\because &Z=\sum_F f_6(F) = P(A=1) = 0.1488 + 0.2592 = 0.408 \\[1ex]
\therefore & P(F=0\mid A=1)=\frac{f_6(0)}{Z}=\frac{0.1488}{0.408}\approx0.365 \\[1ex]
\therefore & P(F=1\mid A=1)=\frac{f_6(1)}{Z}=\frac{0.2592}{0.408}\approx0.635
\end{align*}
$$

### Problem 2

####(a)

$$
\begin{align*}
& \text{Let } f_1(D) = P(D), \quad f_2(D,T)=P(T|D), \quad f_3(D,I)=P(I|D) \\[1ex]
\\[1ex]
& f_{4}(D,T,I) = f_1(D)f_2(D,T)f_3(D,I) \\[1ex]
& f_{4}(0,0,0) = f_1(0) \times f_2(0,0) \times f_3(0,0) = 0.75 \times 0.85 \times 0.80 = 0.510 \\[1ex]
& f_{4}(0,0,1) = f_1(0) \times f_2(0,0) \times f_3(0,1) = 0.75 \times 0.85 \times 0.20 = 0.1275 \\[1ex]
& f_{4}(0,1,0) = f_1(0) \times f_2(0,1) \times f_3(0,0) = 0.75 \times 0.15 \times 0.80 = 0.090 \\[1ex]
& f_{4}(0,1,1) = f_1(0) \times f_2(0,1) \times f_3(0,1) = 0.75 \times 0.15 \times 0.20 = 0.0225 \\[1ex]
& f_{4}(1,0,0) = f_1(1) \times f_2(1,0) \times f_3(1,0) = 0.25 \times 0.15 \times 0.20 = 0.0075 \\[1ex]
& f_{4}(1,0,1) = f_1(1) \times f_2(1,0) \times f_3(1,1) = 0.25 \times 0.15 \times 0.80 = 0.030 \\[1ex]
& f_{4}(1,1,0) = f_1(1) \times f_2(1,1) \times f_3(1,0) = 0.25 \times 0.85 \times 0.20 = 0.0425 \\[1ex]
& f_{4}(1,1,1) = f_1(1) \times f_2(1,1) \times f_3(1,1) = 0.25 \times 0.85 \times 0.80 = 0.170 \\[1ex]
\\[1ex]
& f_5(T,I) = \sum\nolimits_Df_{4}(D,T,I)=P(T,I)\\[1ex]
& f_5(1,0) = f_4(0,1,0) + f_4(1,1,0) = 0.090 + 0.0425 = 0.1325 \\[1ex]
\\[1ex]
& P(D=1|T=1,I=0) = \frac{f_4(1,1,0)}{f_5(1,0)} = \frac{0.0425}{0.1325} \approx 0.321
\end{align*}
$$

#### (b)

$$
\begin{align*}
& f_5(1,1) = f_4(0,1,1) + f_4(1,1,1) = 0.0225 + 0.170 = 0.1925 \\[1ex]
\\[1ex]
& f_6(D,I) = f_1(D)f_3(D,I)=P(D,I) \\[1ex]
& f_6(0,0) = f_1(0)f_3(0,0) = 0.75 \times 0.80 = 0.600 \\[1ex]
& f_6(0,1) = f_1(0)f_3(0,1) = 0.75 \times 0.20 = 0.150 \\[1ex]
& f_6(1,0) = f_1(1)f_3(1,0) = 0.25 \times 0.20 = 0.050 \\[1ex]
& f_6(1,1) = f_1(1)f_3(1,1) = 0.25 \times 0.80 = 0.200 \\[1ex]
\\[1ex]
& f_7(I) = \sum\nolimits_D f_6(D,I) = P(I) \\[1ex]
& f_7(1) = f_6(0,1) + f_6(1,1) = 0.150 + 0.200 = 0.350 \\[1ex]
\\[1ex]
& P(T=1|I=1) = \frac{f_5(1,1)}{f_7(1)} = \frac{0.1925}{0.350} = 0.550
\end{align*}
$$





### Problem 3

#### (a)

$$
\begin{align*}
& \text{Let } f_1(H) = P(H), \quad f_2(H,B)=P(B|H), \quad f_3(H,U)=P(U|H), \quad f_4(B,U,N)=P(N|B,U) \\[1ex]
\\[1ex]
& f_5(H,B,U) = f_1(H)f_2(H,B)f_3(H,U) = P(H,B,U)\\[1ex]
& f_5(0,0,0) = f_1(0)f_2(0,0)f_3(0,0) = 0.4 \times 0.8 \times 0.75 = 0.24 \\[1ex]
& f_5(0,0,1) = f_1(0)f_2(0,0)f_3(0,1) = 0.4 \times 0.8 \times 0.25 = 0.08 \\[1ex]
& f_5(0,1,0) = f_1(0)f_2(0,1)f_3(0,0) = 0.4 \times 0.2 \times 0.75 = 0.06 \\[1ex]
& f_5(0,1,1) = f_1(0)f_2(0,1)f_3(0,1) = 0.4 \times 0.2 \times 0.25 = 0.02 \\[1ex]
& f_5(1,0,0) = f_1(1)f_2(1,0)f_3(1,0) = 0.6 \times 0.4 \times 0.25 = 0.06 \\[1ex]
& f_5(1,0,1) = f_1(1)f_2(1,0)f_3(1,1) = 0.6 \times 0.4 \times 0.75 = 0.18 \\[1ex]
& f_5(1,1,0) = f_1(1)f_2(1,1)f_3(1,0) = 0.6 \times 0.6 \times 0.25 = 0.09 \\[1ex]
& f_5(1,1,1) = f_1(1)f_2(1,1)f_3(1,1) = 0.6 \times 0.6 \times 0.75 = 0.27 \\[1ex]
\\[1ex]
& f_6(B,U) = \sum\nolimits_Hf_5(H,B,U)=P(B,U)\\[1ex]
& f_6(0,0) = f_5(0,0,0) + f_5(1,0,0) = 0.24 + 0.06 = 0.30 \\[1ex]
& f_6(0,1) = f_5(0,0,1) + f_5(1,0,1) = 0.08 + 0.18 = 0.26 \\[1ex]
& f_6(1,0) = f_5(0,1,0) + f_5(1,1,0) = 0.06 + 0.09 = 0.15 \\[1ex]
& f_6(1,1) = f_5(0,1,1) + f_5(1,1,1) = 0.02 + 0.27 = 0.29 \\[1ex]
\\[1ex]
& f_7(B,U,N) = f_4(B,U,N)f_6(B,U) = P(B,U,N)\\[1ex]
& f_7(0,0,0) = f_4(0,0,0)f_6(0,0) = 1.00 \times 0.30 = 0.30 \\[1ex]
& f_7(0,0,1) = f_4(0,0,1)f_6(0,0) = 0.00 \times 0.30 = 0.00 \\[1ex]
& f_7(0,1,0) = f_4(0,1,0)f_6(0,1) = 0.20 \times 0.26 = 0.052 \\[1ex]
& f_7(0,1,1) = f_4(0,1,1)f_6(0,1) = 0.80 \times 0.26 = 0.208 \\[1ex]
& f_7(1,0,0) = f_4(1,0,0)f_6(1,0) = 0.30 \times 0.15 = 0.045 \\[1ex]
& f_7(1,0,1) = f_4(1,0,1)f_6(1,0) = 0.70 \times 0.15 = 0.105 \\[1ex]
& f_7(1,1,0) = f_4(1,1,0)f_6(1,1) = 0.05 \times 0.29 = 0.0145 \\[1ex]
& f_7(1,1,1) = f_4(1,1,1)f_6(1,1) = 0.95 \times 0.29 = 0.2755 \\[1ex]

\\[1ex]
& f_8(N) = \sum\nolimits_{B,U}f_7(B,U,N)=P(N)\\[1ex]
& f_8(0) = f_7(0,0,0) + f_7(0,1,0) + f_7(1,0,0) + f_7(1,1,0) = 0.30 + 0.052 + 0.045 + 0.0145 = 0.4115 \\[1ex]
& f_8(1) = f_7(0,0,1) + f_7(0,1,1) + f_7(1,0,1) + f_7(1,1,1) = 0.00 + 0.208 + 0.105 + 0.2755 = 0.5885
\end{align*}
$$

#### (b)

Because rejection sampling only accepts samples that satisfy \(N=1\), so the acceptance probability is equal to $P(N=1)\approx0.589$.

#### (c)

$$
\begin{align*}
\hat P(B=1\mid N=1) &= \frac{\text{Accepted samples with }B=1}{\text{Total accepted samples}}\\
&=\frac{5}{12} \approx 0.417
\end{align*}
$$

### Problem 4

#### (a)

$$
\begin{align*}
\widehat{P}(B=1 | H=1, N=1) &= \frac{\sum_i w_i \mathbf{1}[B_i=1]}{\sum_i w_i} \\[1ex]
&= \frac{w_1 + w_3 + w_5}{w_1 + w_2 + w_3 + w_4 + w_5} \\[1ex]
&= \frac{0.420 + 0.570 + 0.570}{0.420 + 0.480 + 0.570 + 0.000 + 0.570} \\[1ex]
&= \frac{1.560}{2.040} \approx 0.765
\end{align*}
$$

#### (b)

$$
\begin{align*}
ESS &= \frac{(\sum_i w_i)^2}{\sum_i w_i^2} \\[1ex]
&= \frac{(0.420 + 0.480 + 0.570 + 0.000 + 0.570)^2}{0.420^2 + 0.480^2 + 0.570^2 + 0.000^2 + 0.570^2} \\[1ex]
&= \frac{(2.040)^2}{0.1764 + 0.2304 + 0.3249 + 0.0000 + 0.3249} \\[1ex]
&= \frac{4.1616}{1.0566} \approx 3.94
\end{align*}
$$

### Problem 5

#### (a)

$$
\begin{align*}
& \text{Let } h(B,H,U,N) = P(B|H)P(N|B,U)\\[1ex]
& \because P(B\mid H,U,N)\propto P(B\mid H)P(N\mid B,U)\\[1ex]
& \therefore h(1,0,1,1) = P(B=1|H=0)P(N=1|B=1,U=1) = 0.2 \times 0.95 = 0.19\\[1ex]
& \therefore h(0,0,1,1) = P(B=0|H=0)P(N=1|B=0,U=1) = 0.8 \times 0.80 = 0.64\\[1ex]
& \therefore P(B=1|H=0,U=1,N=1) = \frac{h(1,0,1,1)}{h(1,0,1,1)+h(0,0,1,1)} = \frac{0.19}{0.19+0.64} \approx 0.229
\end{align*}
$$

#### (b)

In Gibbs Sampling, \(B\) is resampled from its conditional distribution given the current values of \(H\), \(U\), and \(N\). Therefore, the probability that \(B\) flips from 0 to 1 is the same as the probability calculated in part (a), which is 0.229.

#### (c)

$$
\begin{align*}
& \text{From above:} \\[1ex]
& f_6(B,U) = \sum\nolimits_Hf_5(H,B,U)=P(B,U)\\[1ex]
& f_6(0,0) = 0.30 \quad f_6(0,1) = 0.26 \\[1ex]
& f_6(1,0) = 0.15 \quad f_6(1,1) = 0.29 \\[1ex]
& \text{Then:} \\[1ex]
& P(B=0,U=1) = 0.26 \quad P(B=1,U=1) = 0.29\\[1ex]
& P(B=0|U=1) = \frac{0.26}{0.26+0.29} \approx 0.473\\[1ex]
& P(B=1|U=1) = \frac{0.29}{0.26+0.29} \approx 0.527\\[1ex]
& \text{Therefore:} \\[1ex]
& P(B=1\mid U=1)>P(B=0\mid U=1)
\end{align*}
$$

### Problem 6

#### (a)

$$
\begin{align*}
& P(X_1=M|X_0=I) = 0.2
\end{align*}
$$

#### (b)

$$
\begin{align*}
P(X_2=M|X_0=I) 
&=P(X_1=I|X_0=I)P(X_2=M|X_1=I)\\[1ex]
&\quad+P(X_1=B|X_0=I)P(X_2=M|X_1=B)\\[1ex]
&\quad+P(X_1=M|X_0=I)P(X_2=M|X_1=M)\\[1ex]
&=0.5\times0.2+0.3\times0.2+0.2\times0.7\\[1ex]
&=0.10+0.06+0.14\\[1ex]
&=0.300
\end{align*}
$$

#### (c)

$$
\begin{align*}
&\text{Let the state order be }(I,B,M)\\[1ex]
&P=
\begin{pmatrix}
0.5&0.3&0.2\\
0.2&0.6&0.2\\
0.1&0.2&0.7
\end{pmatrix}\quad
P^2=
\begin{pmatrix}
0.330&0.370&0.300\\
0.240&0.460&0.300\\
0.160&0.290&0.550
\end{pmatrix} \quad
P^3=
\begin{pmatrix}
0.269&0.381&0.350\\
0.242&0.408&0.350\\
0.193&0.332&0.475
\end{pmatrix}\\[1ex]
&\text{Therefore }P(X_3=B|X_0=I)=0.381\\[1ex]
\end{align*}
$$
