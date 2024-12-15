# DP动规学习笔记
![alt text](image.png)
ps: [MarkDown支持高亮的语言点击查看](https://blog.csdn.net/u012102104/article/details/78950290)
### 11/19看到1.1 数字三角形模型
```cpp
    cin >> t;
	while (t--) {
		memset(f, 0, sizeof f);
		memset(p, 0, sizeof p);
		int r, c;
		cin >> r >> c;
		for (int i = 1; i <= r; i++) {
			for (int j = 1; j <= c; j++) {
				cin >> p[i][j];
			}
		}
		for (int i = 1; i <= r; i++) {
			for (int j = 1; j <= c; j++) {
				f[i][j] = max(f[i - 1][j] + p[i][j], f[i][j - 1] + p[i][j]);
			}
		}
		cout << f[r][c] << endl;
	}
```

### 12/2 最长上升子序列模型（1）
**基础模版：**
```cpp
#include<iostream>
#include<algorithm>
using namespace std;
const int N =1010;
int a[N];
int f[N];
int main(){
    //输入
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    //求最长子序列和 & 最长子序列
    int res = 0;
    for (int i = 0; i < n; i++) {
        f[i] = a[i];
        //f[i] = 1;
        for (int j = 0; j < i; j++) {
            if (a[i] > a[j]) {
                f[i] = max(f[i], f[j] + a[i]);
                // f[i] =max (f[i], f[j] + 1)
            }
        }
        res = max(res, f[i]);
    }
    return 0;
}
```

**最长上升子序列O(nlogn)复杂度代码：**
（二分+动态规划）
思路：首先数组a中存输入的数（原本的数），开辟一个数组f用来存结果，最终数组f的长度就是最终的答案；假如数组f现在存了数，当到了数组a的第i个位置时，首先判断a[i] > f[cnt] ？ 若是大于则直接将这个数添加到数组f中，即f[++cnt] = a[i];这个操作时显然的。
当a[i] <= f[cnt] 的时,我们就用a[i]去替代数组f中的第一个大于等于a[i]的数，因为在整个过程中我们维护的数组f 是一个递增的数组，所以我们可以用二分查找在 logn 的时间复杂的的情况下直接找到对应的位置，然后替换，即f[l] = a[i]。

我们用a[i]去替代f[i]的含义是：以a[i]为最后一个数的严格单调递增序列,这个序列中数的个数为l个。

这样当我们遍历完整个数组a后就可以得到最终的结果。

时间复杂度分析：O(nlogn)

```cpp
#include<iostream>

using namespace std;
const int N = 1e5 + 10;
int cnt = 0;
int n;
int a[N], f[N];

int find(int x) {
	int left = 1;
	int right = cnt;
	while (left < right) {
		int mid = (left + right) / 2;
		if (x <= f[mid]) {
			right = mid;
		}
		else left = mid + 1;
	}
	return left;
}


int main() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	f[++cnt] = a[1];
	for (int i = 2; i <= n; i++) {
		if (a[i] > f[cnt])f[++cnt] = a[i];
		else {
			int l = find(a[i]);
			f[l] = a[i];
		}
	}
	cout << cnt;
	return 0;
}
```


**进阶题型——登山**
需要正向求一次最长上升子序列，再反向求一次，最后寻找最大值
![alt text](image-1.png)
```cpp
for (int i = 0; i < n; i++) {
            f[i]=1;
            for (int j = 0; j < i; j++) {
                if (a[i] > a[j]) {
                    f[i] = max(f[i], f[j] + 1);
                }
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            h[i]=1;
            for (int j = n - 1; j > i; j--) {
                if (a[i] > a[j]) {
                    h[i] = max(h[i], h[j] + 1);
                }
            }
        }
        int max1 = 0;
        for (int i = 0; i < n; i++) {
            //注意：此处需要-1是因为双向的交点重合了！！！
            max1 = max(max1, h[i]+f[i]-1);
        }

```


### 12/4 最长上升子序列模型（2）

[题目：导弹拦截](https://www.acwing.com/problem/content/1012/)
```cpp
#include<iostream>
#include<algorithm>

using namespace std;
const int N =1010;

int n;
int a[N];
int f[N];
int h[N];

int main(){
    while(cin>>a[n])n++;
    //for(int i=0;i<n;i++)cout<<a[i];
    
    /*第一问：求最长下降子序列*/
    int ans1= 0;
    for(int i=0;i<n;i++){
        f[i]=1;
        for(int j=0;j<i;j++){
            if(a[i]<=a[j]){
                f[i] = max (f[j]+1,f[i]);
            }
        }
        ans1 = max (ans1,f[i]);
    }
    cout<<ans1<<endl;
    
    /*第二问：求最长下降子序列
    
    
    思路1：（代码略）
    设最长上升子序列长度为l
    所求上升子序列为h
    那么h<=l
    因为最长上升子序列任意两个不在一组内
    (如果在同一个组内，则每个组的数不成为一个不生子序列，矛盾）
    所以l==h
    
    思路2：贪心
    
    */
    int cnt = 0;
    for(int i=0;i<n;i++){
        int k = 0;
        while(k<cnt&&a[i]>h[k])k++;
        if(k == cnt )h[cnt++] = a[i];
        else h[k] = a[i];
        
    }
    cout <<cnt;
    return 0;
}
```

[导弹防御系统](https://www.acwing.com/problem/content/189/)

```cpp
#include<iostream>
#include<algorithm>

using namespace std;
const int N =55;
int a[N];
int up[N];
int down[N];
int n;
int ans;
void dfs (int u,int su,int sd){
    if(su+sd>=ans){
        return ;
    }
    if(u==n){
        ans = min(ans,su+sd);
        return ;
    }
    //下降序列
    int k=0;
    while(k<su&&up[k]>=a[u])k++;
    if(k<su){
        int t =up[k];
        up[k] = a[u];
        dfs(u+1,su,sd);
        up[k] = t;
    }else {
        up[k] = a[u];
        dfs(u+1,su+1,sd);
    }
    
    //上升序列
    k=0;
    while(k<sd&&down[k]<=a[u])k++;
    if(k<sd){
        int t = down[k];
        down[k] = a[u];
        dfs(u+1,su,sd);
        down[k] = t; 
    }else {
        down[k] = a[u];
        dfs(u+1,su,sd+1);
    }
    
}

int main(){
   
    while(cin>>n,n){
        //cin>>n;
        for(int i =0 ;i<n;i++){
            cin>>a[i];
        }
        ans = n ;
        dfs(0,0,0);
        cout<<ans<<endl;
    }
    
    
    return 0;
}
```
[题目：最长上升公共子序列](https://www.acwing.com/problem/content/274/)

![alt text](image-2.png)

```cpp
#include<iostream>

using namespace std;
const int N=3010;
int a[N];
int b[N];
int f[N][N];


int main(){
    int n;
    cin>>n;
    for(int i=1;i<=n;i++)cin>>a[i];
    for(int j=1;j<=n;j++)cin>>b[j];
    
    for(int i=1;i<=n;i++){
        int max1=1;
        for(int j=1;j<=n;j++){
            f[i][j] = f[i-1][j];
            if(a[i]==b[j]){
                f[i][j] = max(max1,f[i][j]);
            }
            if(b[j]<a[i])max1 = max(max1,f[i-1][j]+1);
        }
    }
    int res = 0;
    for(int i=1;i<=n;i++)res =max(res,f[n][i]);
    cout <<res;
    
    return 0;
}
```


### 12/15 背包模型（1）
复习：[背包模型基础模版](https://blog.csdn.net/hehongfei_/article/details/136474781)

![alt text](image-3.png)

**当空间优化成1维之后，只有完全背包问题的体积是从小到大循环的**

[宠物小精灵之收服](https://www.acwing.com/problem/content/1024/)
做01背包问题的时候，要考虑怎么找出体积和价值
![alt text](image-4.png)

```cpp
#include<iostream>

using namespace std;
const int N =1010;
const int M =510;
int n , m, k;
int f[N][M];


int main(){
    cin>>n>>m>>k;//N，M，K，分别代表小智的精灵球数量、皮卡丘初始的体力值、野生小精灵的数量。
    for(int i=0;i<k;i++){
        int v1,v2;
        cin>>v1>>v2;
        for(int j=n;j>=v1;j--){
            for(int s=m;s>v2;s--){
                f[j][s] = max(f[j][s],f[j-v1][s-v2]+1);
            }
        }
    }
    cout<<f[n][m]<<" ";
    
    int x = m;
    while(x>0&&f[n][x]==f[n][m])x--;
    cout<<m-x;
    
    //输出为一行，包含两个整数：C，R，分别表示最多收服C个小精灵，以及收服C个小精灵时皮卡丘的剩余体力值最多为R。
    
    return 0;
}
```

