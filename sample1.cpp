#include <bits/stdc++.h>
#define rep(i,n) for(int i=0;i<(n);i++)
using namespace std;
typedef long long ll;

int main() {
	int n;
	cin>>n;
	vector<string> a(n);
	rep(i,n){
		string q,w;
		cin>>q>>w;
		a[i] = q+w;
	}
	string gara[4] = {"S" ,"H", "C", "D"};
	string num[13] = {"1","2","3","4","5","6","7","8","9","10","11","12","13"};
	for (int i=0;i<4;i++){
		for (int j=0;j<13;j++){
			string test = gara[i] + num[j];
			bool flg = false;
			for (int k=0;k<n;k++){
				if (test == a[k]){
					flg = true;
					break;
				}
			}
			if (flg) continue;
			else{
				cout << gara[i] << " " << num[j] << endl;
			}
		}
	}


}
