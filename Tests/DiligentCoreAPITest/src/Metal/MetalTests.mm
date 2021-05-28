/*
 *  Copyright 2019-2021 Diligent Graphics LLC
 *  
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *      http://www.apache.org/licenses/LICENSE-2.0
 *  
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  In no event and under no legal theory, whether in tort (including negligence), 
 *  contract, or otherwise, unless required by applicable law (such as deliberate 
 *  and grossly negligent acts) or agreed to in writing, shall any Contributor be
 *  liable for any damages, including any direct, indirect, special, incidental, 
 *  or consequential damages of any character arising as a result of this License or 
 *  out of the use or inability to use the software (including but not limited to damages 
 *  for loss of goodwill, work stoppage, computer failure or malfunction, or any and 
 *  all other commercial damages or losses), even if such Contributor has been advised 
 *  of the possibility of such damages.
 */

#include "TestingEnvironment.hpp"
#include "TestingSwapChainBase.hpp"
#include "ShaderMacroHelper.hpp"

#include "gtest/gtest.h"

#include "InlineShaders/MetalTestsMSL.h"
#include "RayTracingTestConstants.hpp"

#if METAL_SUPPORTED
#    include <Metal/Metal.h>
#    include "DeviceContextMtl.h"

namespace Diligent
{
namespace Testing
{

void RayTracingReferenceMtl(ISwapChain* pSwapChain);
void ThreadgroupMemoryReferenceMtl(ISwapChain* pSwapChain);

} // namespace Testing
} // namespace Diligent

using namespace Diligent;
using namespace Diligent::Testing;

namespace
{

static void RayTracingTest(const int Mode)
{
    auto*       pEnv       = TestingEnvironment::GetInstance();
    auto*       pDevice    = pEnv->GetDevice();
    const auto& deviceInfo = pDevice->GetDeviceInfo();
    if (!deviceInfo.IsMetalDevice() || !deviceInfo.Features.RayTracing)
    {
        GTEST_SKIP() << "Ray tracing is not supported by this device";
    }

    auto*       pSwapChain = pEnv->GetSwapChain();
    auto*       pContext   = pEnv->GetDeviceContext();
    const auto& SCDesc     = pSwapChain->GetDesc();

    RefCntAutoPtr<ITestingSwapChain> pTestingSwapChain(pSwapChain, IID_TestingSwapChain);
    if (pTestingSwapChain)
    {
        pContext->Flush();
        pContext->InvalidateState();

        RayTracingReferenceMtl(pSwapChain);
        pTestingSwapChain->TakeSnapshot();
    }

    TestingEnvironment::ScopedReset EnvironmentAutoReset;

    const auto& Vertices = TestingConstants::TriangleClosestHit::Vertices;

    RefCntAutoPtr<IBuffer> pVertexBuffer;
    RefCntAutoPtr<IBuffer> pConstuffer1;
    RefCntAutoPtr<IBuffer> pConstuffer2;
    RefCntAutoPtr<IBuffer> pConstuffer3;
    {
        BufferDesc BuffDesc;
        BuffDesc.Name          = "Triangle vertices";
        BuffDesc.BindFlags     = BIND_RAY_TRACING;
        BuffDesc.uiSizeInBytes = sizeof(Vertices);
        BufferData BuffData{Vertices, sizeof(Vertices)};
        pDevice->CreateBuffer(BuffDesc, &BuffData, &pVertexBuffer);
        ASSERT_NE(pVertexBuffer, nullptr);

        BuffDesc.Name           = "Constants";
        BuffDesc.BindFlags      = BIND_UNIFORM_BUFFER;
        BuffDesc.uiSizeInBytes  = sizeof(float) * 4;
        BuffDesc.Usage          = USAGE_DYNAMIC;
        BuffDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
        pDevice->CreateBuffer(BuffDesc, nullptr, &pConstuffer1);
        ASSERT_NE(pConstuffer1, nullptr);

        pDevice->CreateBuffer(BuffDesc, nullptr, &pConstuffer2);
        ASSERT_NE(pConstuffer2, nullptr);

        BuffDesc.BindFlags         = BIND_SHADER_RESOURCE;
        BuffDesc.Usage             = USAGE_DEFAULT;
        BuffDesc.CPUAccessFlags    = CPU_ACCESS_NONE;
        BuffDesc.Mode              = BUFFER_MODE_STRUCTURED;
        BuffDesc.ElementByteStride = sizeof(float) * 4;
        pDevice->CreateBuffer(BuffDesc, nullptr, &pConstuffer3);
        ASSERT_NE(pConstuffer3, nullptr);

        void* pMapped = nullptr;
        pContext->MapBuffer(pConstuffer1, MAP_WRITE, MAP_FLAG_DISCARD, pMapped);
        const float Const1[] = {0.5f, 0.9f, 0.75f, 1.0f};
        memcpy(pMapped, Const1, sizeof(Const1));
        pContext->UnmapBuffer(pConstuffer1, MAP_WRITE);

        pContext->MapBuffer(pConstuffer2, MAP_WRITE, MAP_FLAG_DISCARD, pMapped);
        const float Const2[] = {0.2f, 0.0f, 1.0f, 0.5f};
        memcpy(pMapped, Const2, sizeof(Const2));
        pContext->UnmapBuffer(pConstuffer2, MAP_WRITE);

        const float Const3[] = {0.9f, 0.1f, 0.2f, 1.0f};
        pContext->UpdateBuffer(pConstuffer3, 0, sizeof(Const3), Const3, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    RefCntAutoPtr<IBottomLevelAS> pBLAS;
    {
        BLASTriangleDesc TriangleDesc;
        TriangleDesc.GeometryName         = "Triangle";
        TriangleDesc.MaxVertexCount       = _countof(Vertices);
        TriangleDesc.VertexValueType      = VT_FLOAT32;
        TriangleDesc.VertexComponentCount = 3;
        TriangleDesc.MaxPrimitiveCount    = 1;

        BLASBuildTriangleData TriangleData;
        TriangleData.GeometryName         = TriangleDesc.GeometryName;
        TriangleData.pVertexBuffer        = pVertexBuffer;
        TriangleData.VertexStride         = sizeof(Vertices[0]);
        TriangleData.VertexCount          = TriangleDesc.MaxVertexCount;
        TriangleData.VertexValueType      = TriangleDesc.VertexValueType;
        TriangleData.VertexComponentCount = TriangleDesc.VertexComponentCount;
        TriangleData.PrimitiveCount       = 1;
        TriangleData.Flags                = RAYTRACING_GEOMETRY_FLAG_OPAQUE;

        BottomLevelASDesc ASDesc;
        ASDesc.Name          = "Triangle BLAS";
        ASDesc.Flags         = RAYTRACING_BUILD_AS_NONE;
        ASDesc.pTriangles    = &TriangleDesc;
        ASDesc.TriangleCount = 1;
        pDevice->CreateBLAS(ASDesc, &pBLAS);
        ASSERT_NE(pBLAS, nullptr);

        BufferDesc BuffDesc;
        BuffDesc.Name          = "BLAS Scratch Buffer";
        BuffDesc.Usage         = USAGE_DEFAULT;
        BuffDesc.BindFlags     = BIND_RAY_TRACING;
        BuffDesc.uiSizeInBytes = pBLAS->GetScratchBufferSizes().Build;
        RefCntAutoPtr<IBuffer> pScratchBuffer;
        pDevice->CreateBuffer(BuffDesc, nullptr, &pScratchBuffer);
        ASSERT_NE(pScratchBuffer, nullptr);

        BuildBLASAttribs Attribs;
        Attribs.pBLAS                       = pBLAS;
        Attribs.BLASTransitionMode          = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.GeometryTransitionMode      = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.pTriangleData               = &TriangleData;
        Attribs.TriangleDataCount           = 1;
        Attribs.pScratchBuffer              = pScratchBuffer;
        Attribs.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        pContext->BuildBLAS(Attribs);
    }

    RefCntAutoPtr<ITopLevelAS> pTLAS;
    {
        TopLevelASDesc TLASDesc;
        TLASDesc.Name             = "TLAS";
        TLASDesc.MaxInstanceCount = 1;
        TLASDesc.Flags            = RAYTRACING_BUILD_AS_NONE;
        pDevice->CreateTLAS(TLASDesc, &pTLAS);
        ASSERT_NE(pTLAS, nullptr);

        TLASBuildInstanceData Instance;
        Instance.InstanceName = "Instance";
        Instance.pBLAS        = pBLAS;
        Instance.Flags        = RAYTRACING_INSTANCE_NONE;

        BufferDesc BuffDesc;
        BuffDesc.Name          = "TLAS Scratch Buffer";
        BuffDesc.Usage         = USAGE_DEFAULT;
        BuffDesc.BindFlags     = BIND_RAY_TRACING;
        BuffDesc.uiSizeInBytes = pTLAS->GetScratchBufferSizes().Build;
        RefCntAutoPtr<IBuffer> pScratchBuffer;
        pDevice->CreateBuffer(BuffDesc, nullptr, &pScratchBuffer);
        ASSERT_NE(pScratchBuffer, nullptr);

        BuffDesc.Name          = "TLAS Instance Buffer";
        BuffDesc.Usage         = USAGE_DEFAULT;
        BuffDesc.BindFlags     = BIND_RAY_TRACING;
        BuffDesc.uiSizeInBytes = TLAS_INSTANCE_DATA_SIZE;
        RefCntAutoPtr<IBuffer> pInstanceBuffer;
        pDevice->CreateBuffer(BuffDesc, nullptr, &pInstanceBuffer);
        ASSERT_NE(pInstanceBuffer, nullptr);

        BuildTLASAttribs Attribs;
        Attribs.pTLAS                        = pTLAS;
        Attribs.pInstances                   = &Instance;
        Attribs.InstanceCount                = 1;
        Attribs.HitGroupStride               = 1;
        Attribs.BindingMode                  = HIT_GROUP_BINDING_MODE_PER_GEOMETRY;
        Attribs.TLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.BLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.pInstanceBuffer              = pInstanceBuffer;
        Attribs.InstanceBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.pScratchBuffer               = pScratchBuffer;
        Attribs.ScratchBufferTransitionMode  = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        pContext->BuildTLAS(Attribs);
    }

    RefCntAutoPtr<IShader> pCS;
    {
        ShaderMacroHelper Macros;
        if (Mode == 2 || Mode == 3)
        {
            // Signature 1
            Macros.AddShaderMacro("CONST_BUFFER_1", 0);
            Macros.AddShaderMacro("TLAS_1",         1);
            // Signature 2
            Macros.AddShaderMacro("CONST_BUFFER_2", 2);
            Macros.AddShaderMacro("CONST_BUFFER_3", 3);
            Macros.AddShaderMacro("TLAS_2",         4);
        }

        ShaderCreateInfo ShaderCI;
        ShaderCI.SourceLanguage  = SHADER_SOURCE_LANGUAGE_MSL;
        ShaderCI.Desc.ShaderType = SHADER_TYPE_COMPUTE;
        ShaderCI.Desc.Name       = "CS";
        ShaderCI.EntryPoint      = "CSMain";
        ShaderCI.Source          = MSL::RayTracing_CS.c_str();
        ShaderCI.Macros          = Macros;

        pDevice->CreateShader(ShaderCI, &pCS);
        ASSERT_NE(pCS, nullptr);
    }

    ComputePipelineStateCreateInfo PSOCreateInfo;

    PSOCreateInfo.PSODesc.PipelineType                       = PIPELINE_TYPE_COMPUTE;
    PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    PSOCreateInfo.pCS                                        = pCS;
    PSOCreateInfo.PSODesc.Name                               = "Metal ray tracing PSO";

    RefCntAutoPtr<IPipelineState>         pPSO;
    RefCntAutoPtr<IShaderResourceBinding> pSRB1;
    RefCntAutoPtr<IShaderResourceBinding> pSRB2;

    if (Mode == 0)
    {
        pDevice->CreateComputePipelineState(PSOCreateInfo, &pPSO);
        ASSERT_NE(pPSO, nullptr);

        pPSO->CreateShaderResourceBinding(&pSRB1);
        ASSERT_NE(pSRB1, nullptr);
    }
    else if (Mode == 1)
    {
        // clang-format off
        const PipelineResourceDesc Resources[]
        {
            {SHADER_TYPE_COMPUTE, "g_Constant1",   1, SHADER_RESOURCE_TYPE_CONSTANT_BUFFER, SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_COMPUTE, "g_TLAS1",       1, SHADER_RESOURCE_TYPE_ACCEL_STRUCT,    SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_COMPUTE, "g_Constant2",   1, SHADER_RESOURCE_TYPE_CONSTANT_BUFFER, SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_COMPUTE, "g_TLAS2",       1, SHADER_RESOURCE_TYPE_ACCEL_STRUCT,    SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_COMPUTE, "g_Constant3",   1, SHADER_RESOURCE_TYPE_BUFFER_SRV,      SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE, PIPELINE_RESOURCE_FLAG_NO_DYNAMIC_BUFFERS},
            {SHADER_TYPE_COMPUTE, "g_ColorBuffer", 1, SHADER_RESOURCE_TYPE_TEXTURE_UAV,     SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}
        };
        // clang-format on
        PipelineResourceSignatureDesc PRSDesc;
        PRSDesc.Name         = "Signature";
        PRSDesc.Resources    = Resources;
        PRSDesc.NumResources = _countof(Resources);

        RefCntAutoPtr<IPipelineResourceSignature> pPRS;
        pDevice->CreatePipelineResourceSignature(PRSDesc, &pPRS);
        ASSERT_NE(pPRS, nullptr);

        IPipelineResourceSignature* Signatures[] = {pPRS};
        PSOCreateInfo.ppResourceSignatures    = Signatures;
        PSOCreateInfo.ResourceSignaturesCount = _countof(Signatures);

        pDevice->CreateComputePipelineState(PSOCreateInfo, &pPSO);
        ASSERT_NE(pPSO, nullptr);

        pPRS->CreateShaderResourceBinding(&pSRB1);
        ASSERT_NE(pSRB1, nullptr);
    }
    else if (Mode == 2 || Mode == 3)
    {
        const auto VarType = (Mode == 2 ? SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE : SHADER_RESOURCE_VARIABLE_TYPE_STATIC);
        // clang-format off
        const PipelineResourceDesc Resources1[]
        {
            {SHADER_TYPE_COMPUTE, "g_Constant1",   1, SHADER_RESOURCE_TYPE_CONSTANT_BUFFER, VarType},
            {SHADER_TYPE_COMPUTE, "g_TLAS1",       1, SHADER_RESOURCE_TYPE_ACCEL_STRUCT,    VarType},
            {SHADER_TYPE_COMPUTE, "g_ColorBuffer", 1, SHADER_RESOURCE_TYPE_TEXTURE_UAV,     VarType}
        };
        const PipelineResourceDesc Resources2[]
        {
            {SHADER_TYPE_COMPUTE, "g_Constant2",   1, SHADER_RESOURCE_TYPE_CONSTANT_BUFFER, VarType},
            {SHADER_TYPE_COMPUTE, "g_TLAS2",       1, SHADER_RESOURCE_TYPE_ACCEL_STRUCT,    VarType},
            {SHADER_TYPE_COMPUTE, "g_Constant3",   1, SHADER_RESOURCE_TYPE_BUFFER_SRV,      VarType}
        };
        // clang-format on
        PipelineResourceSignatureDesc PRSDesc;
        PRSDesc.Name         = "Signature 1";
        PRSDesc.Resources    = Resources1;
        PRSDesc.NumResources = _countof(Resources1);
        PRSDesc.BindingIndex = 0;

        RefCntAutoPtr<IPipelineResourceSignature> pPRS1;
        pDevice->CreatePipelineResourceSignature(PRSDesc, &pPRS1);
        ASSERT_NE(pPRS1, nullptr);

        PRSDesc.Name         = "Signature 2";
        PRSDesc.Resources    = Resources2;
        PRSDesc.NumResources = _countof(Resources2);
        PRSDesc.BindingIndex = 1;

        RefCntAutoPtr<IPipelineResourceSignature> pPRS2;
        pDevice->CreatePipelineResourceSignature(PRSDesc, &pPRS2);
        ASSERT_NE(pPRS2, nullptr);

        IPipelineResourceSignature* Signatures[] = {pPRS1, pPRS2};
        PSOCreateInfo.ppResourceSignatures    = Signatures;
        PSOCreateInfo.ResourceSignaturesCount = _countof(Signatures);

        pDevice->CreateComputePipelineState(PSOCreateInfo, &pPSO);
        ASSERT_NE(pPSO, nullptr);

        if (Mode == 3)
        {
            pPRS1->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_Constant1")->Set(pConstuffer1);
            pPRS1->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS1")->Set(pTLAS);
            pPRS1->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_ColorBuffer")->Set(pTestingSwapChain->GetCurrentBackBufferUAV());

            pPRS2->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_Constant2")->Set(pConstuffer2);
            pPRS2->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_Constant3")->Set(pConstuffer3->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
            pPRS2->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS2")->Set(pTLAS);
        }

        pPRS1->CreateShaderResourceBinding(&pSRB1, true);
        ASSERT_NE(pSRB1, nullptr);

        pPRS2->CreateShaderResourceBinding(&pSRB2, true);
        ASSERT_NE(pSRB2, nullptr);
    }
    else
        UNEXPECTED("Unexpected Mode");

    pContext->SetPipelineState(pPSO);

    if (Mode == 0 || Mode == 1)
    {
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant1")->Set(pConstuffer1);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant2")->Set(pConstuffer2);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant3")->Set(pConstuffer3->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS1")->Set(pTLAS);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS2")->Set(pTLAS);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_ColorBuffer")->Set(pTestingSwapChain->GetCurrentBackBufferUAV());

        pContext->CommitShaderResources(pSRB1, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    else if (Mode == 2)
    {
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant1")->Set(pConstuffer1);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS1")->Set(pTLAS);
        pSRB1->GetVariableByName(SHADER_TYPE_COMPUTE, "g_ColorBuffer")->Set(pTestingSwapChain->GetCurrentBackBufferUAV());

        pSRB2->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant2")->Set(pConstuffer2);
        pSRB2->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constant3")->Set(pConstuffer3->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        pSRB2->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS2")->Set(pTLAS);

        pContext->CommitShaderResources(pSRB1, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        pContext->CommitShaderResources(pSRB2, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    else if (Mode == 3)
    {
        pContext->CommitShaderResources(pSRB1, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        pContext->CommitShaderResources(pSRB2, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    else
        UNEXPECTED("Unexpected Mode");

    DispatchComputeAttribs dispatchAttrs;
    dispatchAttrs.ThreadGroupCountX = (SCDesc.Width + 15) / 16;
    dispatchAttrs.ThreadGroupCountY = (SCDesc.Height + 15) / 16;

    dispatchAttrs.MtlThreadGroupSizeX = 16;
    dispatchAttrs.MtlThreadGroupSizeY = 16;
    dispatchAttrs.MtlThreadGroupSizeZ = 1;

    pContext->DispatchCompute(dispatchAttrs);

    pSwapChain->Present();
}


TEST(MetalTests, RayTracingWithoutPRS)
{
    RayTracingTest(0);
}

TEST(MetalTests, RayTracingWithSinglePRS)
{
    RayTracingTest(1);
}

TEST(MetalTests, RayTracingWithMultiplePRS)
{
    RayTracingTest(2);
}

TEST(MetalTests, RayTracingWithMultiplePRSWithStaticRes)
{
    RayTracingTest(3);
}


TEST(MetalTests, ThreadgroupMemory)
{
    auto*       pEnv       = TestingEnvironment::GetInstance();
    auto*       pDevice    = pEnv->GetDevice();
    const auto& deviceInfo = pDevice->GetDeviceInfo();
    if (!deviceInfo.IsMetalDevice())
        GTEST_SKIP();

    auto*       pSwapChain = pEnv->GetSwapChain();
    auto*       pContext   = pEnv->GetDeviceContext();
    const auto& SCDesc     = pSwapChain->GetDesc();

    RefCntAutoPtr<ITestingSwapChain> pTestingSwapChain(pSwapChain, IID_TestingSwapChain);
    if (pTestingSwapChain)
    {
        pContext->Flush();
        pContext->InvalidateState();

        ThreadgroupMemoryReferenceMtl(pSwapChain);
        pTestingSwapChain->TakeSnapshot();
    }

    TestingEnvironment::ScopedReset EnvironmentAutoReset;

    RefCntAutoPtr<IShader> pCS;
    {
        ShaderCreateInfo ShaderCI;
        ShaderCI.SourceLanguage  = SHADER_SOURCE_LANGUAGE_MSL;
        ShaderCI.Desc.ShaderType = SHADER_TYPE_COMPUTE;
        ShaderCI.Desc.Name       = "CS";
        ShaderCI.EntryPoint      = "CSmain";
        ShaderCI.Source          = MSL::ThreadgroupMemory_CS.c_str();

        pDevice->CreateShader(ShaderCI, &pCS);
        ASSERT_NE(pCS, nullptr);
    }

    ComputePipelineStateCreateInfo PSOCreateInfo;
    PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    PSOCreateInfo.pCS                                        = pCS;

    RefCntAutoPtr<IPipelineState> pPSO;
    pDevice->CreateComputePipelineState(PSOCreateInfo, &pPSO);
    ASSERT_NE(pPSO, nullptr);

    RefCntAutoPtr<IShaderResourceBinding> pSRB;
    pPSO->CreateShaderResourceBinding(&pSRB);
    ASSERT_NE(pSRB, nullptr);

    pSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_OutImage")->Set(pTestingSwapChain->GetCurrentBackBufferUAV());

    const Uint32                     LocalSize = 8;
    RefCntAutoPtr<IDeviceContextMtl> pContextMtl{pContext, IID_DeviceContextMtl};
    pContextMtl->SetComputeThreadgroupMemoryLength(LocalSize * LocalSize * sizeof(float) * 4, 0);

    DispatchComputeAttribs dispatchAttrs;
    dispatchAttrs.ThreadGroupCountX = (SCDesc.Width + LocalSize - 1) / LocalSize;
    dispatchAttrs.ThreadGroupCountY = (SCDesc.Height + LocalSize - 1) / LocalSize;

    dispatchAttrs.MtlThreadGroupSizeX = LocalSize;
    dispatchAttrs.MtlThreadGroupSizeY = LocalSize;
    dispatchAttrs.MtlThreadGroupSizeZ = 1;

    pContext->SetPipelineState(pPSO);
    pContext->CommitShaderResources(pSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    pContext->DispatchCompute(dispatchAttrs);

    pSwapChain->Present();
}

} // namespace

#endif // METAL_SUPPORTED
