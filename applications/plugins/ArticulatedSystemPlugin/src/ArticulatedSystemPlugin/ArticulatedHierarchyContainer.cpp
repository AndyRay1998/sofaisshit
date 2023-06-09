/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <ArticulatedSystemPlugin/ArticulatedHierarchyContainer.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::container
{

// Register in the Factory
int ArticulatedHierarchyContainerClass = core::RegisterObject("This class allow to store and retrieve all the articulation centers from an articulated rigid object")
        .add< ArticulatedHierarchyContainer >()
        ;

// Register in the Factory
int ArticulationCenterClass = core::RegisterObject("This class defines an articulation center. This contains a set of articulations.")
        .add< ArticulationCenter >()
        ;

// Register in the Factory
int ArticulationClass = core::RegisterObject("This class defines an articulation by an axis, an orientation and an index.")
        .add< Articulation >()
        ;

} // namespace sofa::component::container
