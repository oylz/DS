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

#ifndef _ADAPTER_H_
#define _ADAPTER_H_

#include "../matrix.h"
#include "../munkres.h"

template<typename Data, class Container > class Adapter
{
public:
    virtual Matrix<Data> convertToMatrix(const Container &con) const = 0;
    virtual void convertFromMatrix(Container &con, const Matrix<Data> &matrix) const = 0;
    virtual void solve(Container &con)
    {
        auto matrix = convertToMatrix(con);
        m_munkres.solve(matrix);
        convertFromMatrix(con, matrix);
    }
protected:
    Munkres<Data> m_munkres;
};

#endif /* _ADAPTER_H_ */
