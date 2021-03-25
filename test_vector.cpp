#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {

    // 以写模式打开文件
    ofstream outfile;
    outfile.open("../result/sift_hnsw_40_100/test.json");

    // 向文件写入用户输入的数据
    outfile << "{\"a\":1}" << endl;

    // 关闭打开的文件
    outfile.close();
    return 0;
}
