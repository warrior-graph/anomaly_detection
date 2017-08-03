#include <new_eigen.hpp>
#define bug(x) cout << #x << " = " << (x) << '\n'

namespace new_eigen
{
    template <typename T>
    vector<double> get_eigenvals(const T &t)
    {
        vector<double> ans;
        ostringstream os;
        string aux, aux1;
        os << t;
        aux = os.str();
        bool flag = false;
        for(const char &c : aux)
        {
            if(c == '(')
            {
                aux1.clear(), flag = true;
                continue;
            }
            if(c == ',') flag = false, ans.push_back(stod(aux1));
            if(flag) aux1 += c;
        }
        return ans;
    }

    double diss(const Mat &c1, const Mat &c2)
    {
        stringstream ss;
        ostringstream os;
        string s;
        double sum = 0;
        MatrixXd _c1, _c2;
        GeneralizedEigenSolver<MatrixXd> ges;
        cv2eigen(c1,_c1);
        cv2eigen(c2, _c2);
        ges.compute(_c1, _c2);
        vector<double> eigen = new_eigen::get_eigenvals(ges.eigenvalues().transpose());

        for(double i: eigen) sum+= log(i)*log(i);

        return sqrt(sum);
    }
}

