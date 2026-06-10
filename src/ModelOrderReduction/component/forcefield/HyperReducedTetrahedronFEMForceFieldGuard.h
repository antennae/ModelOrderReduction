#pragma once

namespace sofa::component::forcefield
{

constexpr bool isECSWPreparationMethodSupported(
    const bool prepareECSW, const int method, const int largeMethod)
{
    return !prepareECSW || method == largeMethod;
}

} // namespace sofa::component::forcefield

