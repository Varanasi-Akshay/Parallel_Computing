#include <iostream>
#include <vector>

using namespace std;


vector<int> createArray(int n)
{
   vector<int> X(n);
   for (int i=0; i<n; i++)
      X[i] = i*2;

   return X;
}


int main()
{
   vector<int> X1 = createArray(10);
   cout << X1.size() << endl;

   for (int i=0; i< X1.size(); i++)
      cout << X1[i] << " ";

   cout << endl;

   vector<int> X2 = createArray(100);
   cout << X2.size() << endl;

   return 0;
}
