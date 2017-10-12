/*
 *   Copyright (c) 2015 Miroslav Krajicek
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#ifndef _BOOSTMATRIXADAPTER_H_
#define _BOOSTMATRIXADAPTER_H_

#include "adapter.h"
#ifndef WIN32
#include <boost/serialization/array_wrapper.hpp>
#endif
#include <boost/numeric/ublas/matrix.hpp>

template<typename Data> class BoostMatrixAdapter : public Adapter<Data,boost::numeric::ublas::matrix<Data> >
{
public:
    virtual Matrix<Data> convertToMatrix(const boost::numeric::ublas::matrix<Data> &boost_matrix) const override
    {
        const auto rows = boost_matrix.size1 ();
          const auto columns = boost_matrix.size2 ();
          Matrix <Data> matrix (rows, columns);
          for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
              matrix (i, j) = boost_matrix (i, j);
            }
          }
          return matrix;
    }

    virtual void convertFromMatrix(boost::numeric::ublas::matrix<Data> &boost_matrix,const Matrix<Data> &matrix) const override
    {
        const auto rows = matrix.rows();
          const auto columns = matrix.columns();
          for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
              boost_matrix (i, j) = matrix (i, j);
            }
          }
    }
};

#endif /* _BOOSTMATRIXADAPTER_H_ */
