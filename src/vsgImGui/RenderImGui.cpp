/* <editor-fold desc="MIT License">

Copyright(c) 2021 Don Burns, Roland Hill and Robert Osfield.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsgImGui/RenderImGui.h>
#include <vsgImGui/Texture.h>
#include <vsgImGui/implot.h>

#include "../imgui/backends/imgui_impl_vulkan.h"

#include <vsg/app/TransferTask.h>
#include <vsg/core/Array2D.h>
#include <vsg/io/Logger.h>
#include <vsg/maths/color.h>
#include <vsg/state/DescriptorImage.h>
#include <vsg/utils/CoordinateSpace.h>
#include <vsg/vk/Context.h>
#include <vsg/vk/DescriptorPools.h>
#include <vsg/vk/State.h>
#include <vsg/vk/SubmitCommands.h>

#include <cstring>

using namespace vsgImGui;

namespace vsgImGui
{
    void check_vk_result(VkResult err)
    {
        if (err == 0) return;

        vsg::error("[vulkan] Error: VkResult = ", err);
    }

    class ImGuiNode : public vsg::Inherit<vsg::Node, ImGuiNode>
    {
    public:
        ImGuiNode(RenderImGui::LegacyFunction in_func) :
            func(in_func) {}

        RenderImGui::LegacyFunction func;

        void accept(vsg::RecordTraversal&) const override
        {
            func();
        }
    };

    void ImGuiStyle_sRGB_to_linear(ImGuiStyle& style)
    {
        for (size_t i = 0; i < ImGuiCol_COUNT; ++i)
        {
            ImVec4& color = style.Colors[i];
            color = vsg::sRGB_to_linear<float>(color);
        }
    }

} // namespace vsgImGui

RenderImGui::RenderImGui(const vsg::ref_ptr<vsg::Window>& window, bool useClearAttachments)
{
    _init(window, useClearAttachments);
}

RenderImGui::RenderImGui(vsg::ref_ptr<vsg::Device> device, uint32_t queueFamily,
                         vsg::ref_ptr<vsg::RenderPass> renderPass,
                         uint32_t minImageCount, uint32_t imageCount,
                         VkExtent2D imageSize, bool useClearAttachments)
{
    _init(device, queueFamily, renderPass, minImageCount, imageCount, imageSize, useClearAttachments);
}

RenderImGui::~RenderImGui()
{
    // Release our atlas textures before the backend shuts down. We never set BackendUserData,
    // so the backend's own texture cleanup only marks them destroyed.
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
    {
        if (_textures.count(tex->UniqueID) == 0) continue;
        tex->SetTexID(ImTextureID_Invalid);
    }
    _textures.clear();
    _textureContext = {};

    ImGui_ImplVulkan_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

void RenderImGui::add(const LegacyFunction& legacyFunc)
{
    addChild(ImGuiNode::create(legacyFunc));
}

void RenderImGui::_init(const vsg::ref_ptr<vsg::Window>& window, bool useClearAttachments)
{
    auto device = window->getOrCreateDevice();
    auto physicalDevice = device->getPhysicalDevice();

    uint32_t queueFamily = 0;
    std::tie(queueFamily, std::ignore) = physicalDevice->getQueueFamily(window->traits()->queueFlags, window->getSurface());
    auto queue = device->getQueue(queueFamily);

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(*physicalDevice,
                                              *(window->getSurface()),
                                              &capabilities);
    uint32_t imageCount = 3;
    imageCount =
        std::max(imageCount,
                 capabilities.minImageCount); // Vulkan spec requires
                                              // minImageCount to be 1 or greater
    if (capabilities.maxImageCount > 0)
        imageCount = std::min(
            imageCount,
            capabilities.maxImageCount); // Vulkan spec specifies 0 as being
                                         // unlimited number of images

    _init(device, queueFamily, window->getOrCreateRenderPass(), capabilities.minImageCount, imageCount, window->extent2D(), useClearAttachments);
}

void RenderImGui::_init(
    vsg::ref_ptr<vsg::Device> device, uint32_t queueFamily,
    vsg::ref_ptr<vsg::RenderPass> renderPass,
    uint32_t minImageCount, uint32_t imageCount,
    VkExtent2D imageSize, bool useClearAttachments)
{
    IMGUI_CHECKVERSION();

    if (!ImGui::GetCurrentContext())
    {
        ImGui::CreateContext();
    }

    bool sRGB = false;
    for (auto& attachment : renderPass->attachments)
    {
        if (attachment.finalLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR || attachment.finalLayout==VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            if (attachment.format == VK_FORMAT_B8G8R8_SRGB ||
                attachment.format == VK_FORMAT_B8G8R8A8_SRGB ||
                attachment.format == VK_FORMAT_R8G8B8_SRGB ||
                attachment.format == VK_FORMAT_R8G8B8A8_SRGB) sRGB = true;
        }
    }

    if (sRGB)
    {
        ImGuiStyle_sRGB_to_linear(ImGui::GetStyle());
    }

    if (!ImPlot::GetCurrentContext()) ImPlot::CreateContext();

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    for (auto& attachment : renderPass->attachments)
    {
        if (attachment.samples > samples) samples = attachment.samples;
    }

    // ImGui may change this later, but ensure the display
    // size is set to something, to prevent assertions
    // in ImGui::newFrame.
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = static_cast<float>(imageSize.width);
    io.DisplaySize.y = static_cast<float>(imageSize.height);

    _device = device;
    _queueFamily = queueFamily;
    _queue = _device->getQueue(_queueFamily);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *(_device->getInstance());
    init_info.PhysicalDevice = *(_device->getPhysicalDevice());
    init_info.Device = *(_device);
    init_info.QueueFamily = _queueFamily;
    // The backend submits font atlas uploads directly to this queue without VSG's queue mutex. accept() handles
    // those uploads through VSG's TransferTask instead, so the backend only uses the queue if no TransferTask exists.
    init_info.Queue = *(_queue);
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.PipelineInfoMain.MSAASamples = samples;

    // Create Descriptor Pool
    vsg::DescriptorPoolSizes pool_sizes = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    uint32_t maxSets = static_cast<uint32_t>(1000 * pool_sizes.size());
    _descriptorPool = vsg::DescriptorPool::create(_device, maxSets, pool_sizes);

    init_info.DescriptorPool = *_descriptorPool;
    init_info.PipelineInfoMain.RenderPass = *renderPass;
    init_info.Allocator = nullptr;
    init_info.MinImageCount = std::max(minImageCount, 2u); // ImGui's Vulkan backend has an assert that requires MinImageCount to be 2 or more.
    init_info.ImageCount = imageCount;
    init_info.CheckVkResultFn = check_vk_result;
    _imageCount = imageCount;

    ImGui_ImplVulkan_Init(&init_info);

    if (useClearAttachments)
    {
        // clear the depth buffer before view2 gets rendered
        VkClearValue clearValue{};
        clearValue.depthStencil = {1.0f, 0};
        VkClearAttachment attachment{VK_IMAGE_ASPECT_DEPTH_BIT, 1, clearValue};
        VkClearRect rect{VkRect2D{VkOffset2D{0, 0}, VkExtent2D{imageSize.width, imageSize.height}}, 0, 1};
        _clearAttachments = vsg::ClearAttachments::create(vsg::ClearAttachments::Attachments{attachment}, vsg::ClearAttachments::Rects{rect});
    }
}

void RenderImGui::accept(vsg::RecordTraversal& rt) const
{
    auto& commandBuffer = *(rt.getState()->_commandBuffer);
    if (_device.get() != commandBuffer.getDevice()) return;

    // record all the ImGui commands to ImDrawData container
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    // traverse children
    traverse(rt);

    ImGui::EndFrame();
    ImGui::Render();

    // if ImDrawData has been recorded then we need to clear the frame buffer and do the final record to Vulkan command buffer.
    ImDrawData* draw_data = ImGui::GetDrawData();

    // Honour texture requests every frame, even without draw commands, so destroys are not deferred indefinitely.
    // When handled here, hide them from the backend so it doesn't submit to the queue itself.
    if (draw_data && _updateTextures(draw_data, commandBuffer.deviceID)) draw_data->Textures = nullptr;

    if (draw_data && draw_data->CmdLists.size() > 0)
    {
        if (_clearAttachments) _clearAttachments->record(commandBuffer);

        ImGui_ImplVulkan_RenderDrawData(draw_data, &(*commandBuffer));
    }
}

bool RenderImGui::_updateTextures(ImDrawData* drawData, uint32_t deviceID) const
{
    if (!drawData->Textures) return false;

    // Without a TransferTask (no RecordAndSubmitTask) fall back to the backend's own upload path
    auto transferTask = _device->transferTask.ref_ptr();
    if (!transferTask) return false;

    if (!_textureContext)
    {
        _textureContext = vsg::Context::create(_device.get());
        _textureContext->transferTask = transferTask;
        _textureContext->graphicsQueue = _queue;

        // A private pool keeps allocation off the DescriptorPools shared with compile threads, which is not locked
        _textureContext->descriptorPools = vsg::DescriptorPools::create(_device);
    }

    // Copy the given region of ImGui's pixels into the VSG image, expanding Alpha8 to white + alpha
    auto copyPixels = [](ImTextureData* tex, vsg::ubvec4Array2D& image, int x0, int y0, int w, int h) {
        for (int y = y0; y < y0 + h; ++y)
        {
            auto dest = static_cast<uint8_t*>(image.dataPointer(static_cast<size_t>(x0) + static_cast<size_t>(y) * image.width()));
            auto src = static_cast<const uint8_t*>(tex->GetPixelsAt(x0, y));
            if (tex->Format == ImTextureFormat_RGBA32)
            {
                std::memcpy(dest, src, static_cast<size_t>(w) * 4);
                continue;
            }

            for (int x = 0; x < w; ++x)
            {
                dest[x * 4] = dest[x * 4 + 1] = dest[x * 4 + 2] = 255;
                dest[x * 4 + 3] = src[x];
            }
        }
    };

    for (ImTextureData* tex : *drawData->Textures)
    {
        if (tex->Status == ImTextureStatus_OK) continue;

        if (tex->Status == ImTextureStatus_WantCreate)
        {
            // Uploaded by the late transfer after this record traversal, in time for this frame's submit
            auto image = vsg::ubvec4Array2D::create(tex->Width, tex->Height, vsg::Data::Properties{VK_FORMAT_R8G8B8A8_UNORM});
            image->properties.dataVariance = vsg::DYNAMIC_DATA_TRANSFER_AFTER_RECORD;
            copyPixels(tex, *image, 0, 0, tex->Width, tex->Height);
            image->dirty();

            // No mipmaps: the atlas is sampled 1:1
            auto sampler = vsg::Sampler::create();
            sampler->maxLod = 0.0f;
            sampler->addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler->addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler->addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            auto texture = Texture::create(image, sampler);
            texture->compile(*_textureContext);
            if (_textureContext->record()) _textureContext->waitForCompletion();

            _textures[tex->UniqueID] = texture;
            tex->SetTexID(texture->id(deviceID));
            tex->SetStatus(ImTextureStatus_OK);
            continue;
        }

        auto itr = _textures.find(tex->UniqueID);

        if (tex->Status == ImTextureStatus_WantUpdates)
        {
            if (itr == _textures.end()) continue;

            auto image = itr->second->descriptorSet->descriptors.front().cast<vsg::DescriptorImage>()->imageInfoList.front()->imageView->image->data.cast<vsg::ubvec4Array2D>();
            const ImTextureRect& r = tex->UpdateRect;
            copyPixels(tex, *image, r.x, r.y, r.w, r.h);
            image->dirty();
            tex->SetStatus(ImTextureStatus_OK);
            continue;
        }

        // WantDestroy: wait until no in-flight frame can still reference the descriptor set
        if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames >= static_cast<int>(_imageCount))
        {
            if (itr != _textures.end()) _textures.erase(itr);
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }

    return true;
}
