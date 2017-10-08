#ifndef _HUNGARIANOPERH_
#define _HUNGARIANOPERH_
#include "munkres/munkres.h"
#include "munkres/adapters/boostmatrixadapter.h"
#include "Detection.h"

class HungarianOper {
public:
	static Eigen::Matrix<float, -1, 2> Solve(const DYNAMICM &cost_matrix) {
		int rows = cost_matrix.rows();
		int cols = cost_matrix.cols();
		Matrix<double> matrix(rows, cols);
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				matrix(row, col) = cost_matrix(row, col);
			}
		}
		//
		Munkres<double> m;
		m.solve(matrix);

		// 
		std::vector<std::pair<int, int>> pairs;
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				int tmp = (int)matrix(row, col);
				if (tmp == 0) {
					std::pair<int, int> pa;
					pa.first = row;
					pa.second = col;
					pairs.push_back(pa);
				}
			}
		}
		//
		int count = pairs.size();
		Eigen::Matrix<float, -1, 2> re(count, 2);
		for (int i = 0; i < count; i++) {
			std::pair<int, int> &pa = pairs[i];
			re(i, 0) = pa.first;
			re(i, 1) = pa.second;
		}
		return re;
	}
};

#endif