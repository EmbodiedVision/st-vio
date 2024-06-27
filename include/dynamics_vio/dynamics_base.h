/**
This file is part of ST-VIO.

Copyright (c) 2024, Max Planck Gesellschaft.
Developed by Haolong Li <haolong.li at tue dot mpg dot de>
Embodied Vision Group, Max Planck Institute for Intelligent Systems.
If you use this code, please cite the respective publication as
listed in README.md.

ST-VIO is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ST-VIO is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ST-VIO.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "dynamics_vio/utils/utils.h"

namespace dynvio {

class DynamicsBase{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<DynamicsBase>;

    DynamicsBase(int64_t start_t_ns, int64_t end_t_ns):
        start_t_ns(start_t_ns), end_t_ns(end_t_ns){
    }

    virtual ~DynamicsBase(){}

    int64_t get_start_t_ns() const { return start_t_ns; }
    int64_t get_end_t_ns() const { return end_t_ns; }

protected:
    int64_t start_t_ns;
    int64_t end_t_ns;
};
} // namespace dynvio
