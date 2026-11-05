Name: Wan Yiwei 		Student ID: 1155261963


#### Problem 1

##### (a)

$$
\begin{align*}
P(X_2=B)
&=P(X_1=B)P(X_2=B|X_1=B)+P(X_1=Q)P(X_2=B|X_1=Q)\\[1ex]
&=1 \times 0.8 + 0 \times 0.4 = 0.8\\[1ex]
P(X_2=Q)
&=P(X_1=B)P(X_2=Q|X_1=B)+P(X_1=Q)P(X_2=Q|X_1=Q)\\[1ex]
&=1 \times 0.2 + 0 \times 0.6 =0.2 \\[1ex]
\end{align*}
$$

##### (b)

$$
\begin{align*}
P(X_3)
&=P(X_2)T
=
\begin{pmatrix}
0.8&0.2
\end{pmatrix}
\begin{pmatrix}
0.8&0.2\\
0.4&0.6
\end{pmatrix}\\[1ex]
&=
\begin{pmatrix}
0.8 * 0.8 + 0.2 * 0.4 & 0.8 * 0.2 + 0.2 * 0.6
\end{pmatrix}\\[1ex]
&=
\begin{pmatrix}
0.72&0.28
\end{pmatrix}\\[1ex]
\end{align*}
$$

##### (c)

$$
\begin{align*}
\text{Stationarity:}\quad
&\pi T = \pi \\[1ex]
\text{Normalization:}\quad
&\pi_B+\pi_Q=1 \\[1ex]
\text{Let}\quad
&\pi_B=x,\quad \pi_Q=1-x \\[1ex]
\text{Then}\quad
&\begin{pmatrix}x&1-x\end{pmatrix}
\begin{pmatrix}
0.8&0.2\\
0.4&0.6
\end{pmatrix}
=
\begin{pmatrix}
0.8x+0.4(1-x)&0.2x+0.6(1-x)
\end{pmatrix}\\[1ex]
&=
\begin{pmatrix}
0.4+0.4x&0.6-0.4x
\end{pmatrix}
=
\begin{pmatrix}x&1-x\end{pmatrix}\\[1ex]
\text{Therefore}\quad
&
\begin{cases}
0.4+0.4x=x\\
0.6-0.4x=1-x
\end{cases}
\to
x=\frac{2}{3}
\to
\pi=
\begin{pmatrix}
\frac{2}{3}&\frac{1}{3}
\end{pmatrix}
\end{align*}
$$

#### Problem 2

##### (a)

$$
\begin{align*}
\text{Prediction:}\,
& P(X_1 = O) = P(X_0 = O)P(X_1 = O | X_0 = O) + P(X_0 = V)P(X_1 = O | X_0 = V) = 0.4 \times 0.8 + 0.6 \times 0.3 = 0.50 \\
& P(X_1 = V) = P(X_0 = O)P(X_1 = V | X_0 = O) + P(X_0 = V)P(X_1 = V | X_0 = V) = 0.4 \times 0.2 + 0.6 \times 0.7 = 0.50 \\
\\
\text{Observation:}\,
& P(E_1 = +, X_1 = O) = P(E_1 = + | X_1 = O) P(X_1 = O) = 0.75 \times 0.50 = 0.375\\[1ex]
& P(E_1 = +, X_1 = V) = P(E_1 = + | X_1 = V) P(X_1 = V) = 0.25 \times 0.50 = 0.125\\[1ex]
\\
\text{Normalization:}\,
& P(E_1 = +) = P(E_1 = +, X_1 = O) + P(E_1 = +, X_1 = V) = 0.375 + 0.125 = 0.50\\[1ex]
\\
\text{Belief:}\,
& B_1(O) = P(X_1 = O|E_1 = +) = P(E_1 = +, X_1 = O) / P(E_1 = +) = 0.375 / 0.50 = 0.75\\[1ex]
& B_1(V) = P(X_1 = V|E_1 = +) = P(E_1 = +, X_1 = V) / P(E_1 = +) = 0.125 / 0.50 = 0.25\\[1ex]
& B_1(x) = \begin{pmatrix}0.75&0.25\end{pmatrix}
\end{align*}
$$

##### (b)

$$
\begin{align*}
\text{Prediction:}\,
& P(X_2 = O | E_1 = +) = P(X_1 = O|E_1=+) P(X_2 = O | X_1 = O) + P(X_1 = V|E_1=+) P(X_2 = O | X_1 = V) 
\\& = 0.75 \times 0.8 + 0.25 \times 0.3 = 0.675 \\[1ex]
& P(X_2 = V | E_1 = +) = P(X_1 = V|E_1=+) P(X_2 = V | X_1 = V) + P(X_1 = O|E_1=+) P(X_2 = V | X_1 = O)
\\& = 0.25 \times 0.7 + 0.75 \times 0.2 = 0.325 \\[1ex]
\\
\text{Observation:}\,
& P(E_2 = -, X_2 = O | E_1 = +) = P(E_2 = - | X_2 = O) P(X_2 = O | E_1 = +) = 0.25 \times 0.675 = 0.16875\\[1ex]
& P(E_2 = -, X_2 = V | E_1 = +) = P(E_2 = - | X_2 = V) P(X_2 = V | E_1 = +) = 0.75 \times 0.325 = 0.24375\\[1ex]
\\
\text{Normalization:}\,
& P(E_2 = -| E_1 = +) = P(E_2 = -, X_2 = O| E_1 = +) + P(E_2 = -, X_2 = V| E_1 = +) = 0.16875 + 0.24375 = 0.4125\\[1ex]
\\
\text{Belief:}\,
& B_2(O) = P(X_2 = O|E_1 = +, E_2 = -) = P(E_2 = -, X_2 = O | E_1 = +) / P(E_2 = -| E_1 = +) = 0.16875/0.4125\approx 0.4091 \\[1ex]
& B_2(V) = P(X_2 = V|E_1 = +, E_2 = -) = P(E_2 = -, X_2 = V | E_1 = +) / P(E_2 = -| E_1 = +) = 0.24375/0.4125\approx 0.5909 \\[1ex]
& B_2(x) = \begin{pmatrix} 0.4091 & 0.5909 \end{pmatrix}
\end{align*}
$$

#### Problem 3

##### (a)

**First-Order Markov Property** $P(X_t\mid X_{0:t-1})=P(X_t\mid X_{t-1})$: Given the hidden state at the previous moment, the current hidden state is conditionally independent of the hidden states at any earlier moments.

**Observation Independence Property** $P(E_t\mid X_{0:T},E_{1:t-1})=P(E_t\mid X_t)$ : Given the current hidden state, the current observation is conditionally independent of the past observation conditions and any other hidden states.

##### (b)

No. $E_1 \leftarrow X_1 \rightarrow X_2 \rightarrow E_2$ contains a fork at $X_1$ and a chain at $X_2$. Since $X_1$ and $X_2$ are not conditioned on, the path remains active.

Therefore, $E_1$ and $E_2$ are not d-separated, so the graph does not imply $E_1 \perp E_2$.

##### (c)

Yes. $E_1 \leftarrow X_1 \rightarrow X_2 \rightarrow E_2$ contains a fork at $X_1$. Since $X_1$ is conditioned on, the path is blocked.

Therefore, $E_1$ and $E_2$ are d-separated given $X_1$, so the graph implies $E_1 \perp E_2 \mid X_1$.

#### Problem 4

##### (a)

In the Elapse step, each particle is independently sampled to a new location according to the transition model $P(X_{t+1}\mid X_t)$. The resulting particles approximate the predicted state distribution at time $t+1$. No new observation is incorporated during this step.

##### (b)

$$
\begin{align*}
\because & E[N_s]=N\times P(X_{t+1}=s\mid X_t=(4,2))\\[1ex]
\therefore & E[N_{(4,3)}]=20\times0.60=12,\quad E[N_{(4,2)}]=20\times0.25=5,\quad E[N_{(3,2)}]=20\times0.15=3
\end{align*}
$$

#### Problem 5

##### (a)

$$
\begin{align*}
& w_1 = P(r=+|X=A) = 0.8 \quad w_2 = P(r=+|X=B) = 0.5 \quad w_3 = P(r=+|X=A) = 0.8 \quad w_4 = P(r=+|X=C) = 0.3 \\[1ex]
& w_5 = P(r=+|X=D) = 0.6 \quad w_6 = P(r=+|X=E) = 0.1 \quad w_7 = P(r=+|X=B) = 0.5 \quad w_8 = P(r=+|X=F) = 0.3 \\[1ex]
& w_9 = P(r=+|X=C) = 0.3 \quad w_{10} = P(r=+|X=A) = 0.8 \\[1ex]
& W = \sum\nolimits^{10}_{i=1}w_i = 0.8+0.5+0.8+0.3+0.6+0.1+0.5+0.3+0.3+0.8=5.0
\end{align*}
$$

##### (b)

$$
\begin{alignat*}{3}
P(A)&=\frac{w_1+w_3+w_{10}}{W}=\frac{0.8+0.8+0.8}{5.0}=0.48 &\quad
P(B)&=\frac{w_2+w_7}{W}=\frac{0.5+0.5}{5.0}=0.20 &\quad
P(C)&=\frac{w_4+w_9}{W}=\frac{0.3+0.3}{5.0}=0.12\\[1ex]
P(D)&=\frac{w_5}{W}=\frac{0.6}{5.0}=0.12 &\quad
P(E)&=\frac{w_6}{W}=\frac{0.1}{5.0}=0.02 &\quad
P(F)&=\frac{w_8}{W}=\frac{0.3}{5.0}=0.06
\end{alignat*}
$$

#### Problem 6

##### (a)

When the state space is very large or continuous, particle filtering is preferred over exact HMM filtering because exact HMM filtering requires maintaining and updating the probability distribution over all possible states, which can be computationally expensive. For example, in robot  physical localization, there may be an enormous number of possible positions. Particle filtering can instead approximate the probability distribution using a limited number of particles, reducing computational cost at the expense of some accuracy.

##### (b)

$$
\begin{align*}
B_t(x)
&=P(X_t=x\mid e_{1:t})=\frac{P(e_t\mid X_t=x,e_{1:t-1})P(X_t=x\mid e_{1:t-1})}
{P(e_t\mid e_{1:t-1})}\\[2ex]
&=\frac{P(e_t\mid X_t=x)P(X_t=x\mid e_{1:t-1})}
{P(e_t\mid e_{1:t-1})} = \frac{P(e_t\mid X_t=x)P(X_t=x\mid e_{1:t-1})}
{\sum_s P(e_t\mid X_t=s)P(X_t=s\mid e_{1:t-1})}\\[2ex]
&\approx
\frac{
P(e_t\mid X_t=x)\frac{1}{N}
\sum_{i=1}^{N}\mathbf{1}(S_t^{(i)}=x)
}{
\frac{1}{N}\sum_{i=1}^{N}
P(e_t\mid X_t=S_t^{(i)})
}
=
\frac{\sum_{i=1}^{N}w_t^{(i)}
\mathbf{1}(S_t^{(i)}=x)}
{\sum_{i=1}^{N}w_t^{(i)}}
\end{align*}
$$