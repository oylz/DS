/*
 *   Copyright (c) 2007 John Weaver
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

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <initializer_list>
#include <cstdlib>
#include <ostream>

#define XYZMIN(x, y) (x)<(y)?(x):(y)
#define XYZMAX(x, y) (x)>(y)?(x):(y)

template <class T>
class Matrix {
public:
  Matrix();
  Matrix(const size_t rows, const size_t columns);
  Matrix(const std::initializer_list<std::initializer_list<T>> init);
  Matrix(const Matrix<T> &other);
  Matrix<T> & operator= (const Matrix<T> &other);
  ~Matrix();
  // all operations modify the matrix in-place.
  void resize(const size_t rows, const size_t columns, const T default_value = 0);
  void clear();
  T& operator () (const size_t x, const size_t y);
  const T& operator () (const size_t x, const size_t y) const;
  const T mmin() const;
  const T mmax() const;
  inline size_t minsize() { return ((m_rows < m_columns) ? m_rows : m_columns); }
  inline size_t columns() const { return m_columns;}
  inline size_t rows() const { return m_rows;}

  friend std::ostream& operator<<(std::ostream& os, const Matrix &matrix)
  {
      os << "Matrix:" << std::endl;
      for (size_t row = 0 ; row < matrix.rows() ; row++ )
      {
          for (size_t col = 0 ; col < matrix.columns() ; col++ )
          {
              os.width(8);
              os << matrix(row, col) << ",";
          }
          os << std::endl;
      }
      return os;
  }

private:
  T **m_matrix;
  size_t m_rows;
  size_t m_columns;
};

#ifndef USE_EXPORT_KEYWORD
#include "matrix.cpp"
//#define export /*export*/
#endif

#endif /* !defined(_MATRIX_H_) */

