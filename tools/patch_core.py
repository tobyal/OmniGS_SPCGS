"""Small, auditable additions to the copied OmniGS core; never touches the source tree."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]

if ('semantic_ = optimizable_tensors[6]' in (ROOT/'code/src/gaussian_model.cpp').read_text(encoding='utf-8')
    and 'void FORWARD::setSpcDiagnostics' in (ROOT/'code/cuda_rasterizer/forward.cu').read_text(encoding='utf-8')
    and 'add_executable(spcgs_native' in (ROOT/'code/CMakeLists.txt').read_text(encoding='utf-8')):
    print('Core additions already present; no mutation')
    sys.exit(0)

def change(rel, old, new, count=1):
    p = ROOT / 'code' / rel
    s = p.read_text(encoding='utf-8')
    if new in s:
        return
    assert s.count(old) == count, (rel, s.count(old), old[:80])
    p.write_text(s.replace(old, new), encoding='utf-8')

change('include/gaussian_model.h', '    torch::Tensor xyz_;', '''    // Optional SPC-GS attributes, kept in the same row order as xyz.
    torch::Tensor semantic_, pending_semantic_;
    int64_t spc_added_ = 0, spc_pruned_ = 0;
    torch::Tensor xyz_;''')
change('src/gaussian_model.cpp', '    // get_expon_lr_func', '''    if (semantic_.defined()) {
        optimizer_->add_param_group(std::vector<torch::Tensor>{semantic_});
        optimizer_->param_groups()[6].options().set_lr(0.0025);
    }
    // get_expon_lr_func''')
p = ROOT/'code/src/gaussian_model.cpp'
s = p.read_text(encoding='utf-8')
start, end = s.index('void GaussianModel::prunePoints('), s.index('void GaussianModel::densifyAndSplit(')
part = s[start:end]
part = part.replace('optimizable_tensors(6)', 'optimizable_tensors(semantic_.defined() ? 7 : 6)')
part = part.replace('group_idx < 6', 'group_idx < (semantic_.defined() ? 7 : 6)')
part = part.replace('    this->rotation_ = optimizable_tensors[5];', '    this->rotation_ = optimizable_tensors[5];\n    if (semantic_.defined()) semantic_ = optimizable_tensors[6];')
part = part.replace('    auto valid_points_mask = ~mask;', '    auto valid_points_mask = ~mask;\n    spc_pruned_ += mask.sum().item<int64_t>();')
part = part.replace('    auto& param_groups = this->optimizer_->param_groups();\n    auto& state = this->optimizer_->state();\n    for', '    auto& param_groups = this->optimizer_->param_groups();\n    auto& state = this->optimizer_->state();\n    for')
part = part.replace('        new_rotation\n    };', '''        new_rotation
    };
    spc_added_ += new_xyz.size(0);
    if (semantic_.defined()) {
        TORCH_CHECK(pending_semantic_.defined() && pending_semantic_.size(0)==new_xyz.size(0), "Missing semantic ancestry");
        tensors_dict.push_back(pending_semantic_);
    }''')
s = s[:start]+part+s[end:]
s = s.replace('    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask}).repeat({N});', '    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask}).repeat({N});\n    if (semantic_.defined()) pending_semantic_ = semantic_.index({selected_pts_mask}).repeat({N,1});')
s = s.replace('    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask});', '    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask});\n    if (semantic_.defined()) pending_semantic_ = semantic_.index({selected_pts_mask});')
s = s.replace('torch::ones_like(this->getOpacityActivation() * 0.01)', 'torch::full_like(this->getOpacityActivation(), 0.01)')
p.write_text(s, encoding='utf-8')

change('cuda_rasterizer/forward.h', 'namespace FORWARD\n{', '''namespace FORWARD
{
    // Optional outputs from the ACTUAL compositing loop, -1 for no contribution.
    // One synchronous renderer per process; buffers must live until stream completion.
    void setSpcDiagnostics(int* ids, float* weights);''')
change('cuda_rasterizer/forward.cu', 'template <uint32_t CHANNELS>\n__global__ void __launch_bounds__', '''namespace { thread_local int* spc_ids = nullptr; thread_local float* spc_weights = nullptr; }
void FORWARD::setSpcDiagnostics(int* ids, float* weights) { spc_ids=ids; spc_weights=weights; }

template <uint32_t CHANNELS>
__global__ void __launch_bounds__''', count=2) if False else None
p=ROOT/'code/cuda_rasterizer/forward.cu'
s=p.read_text(encoding='utf-8')
marker='template <uint32_t CHANNELS>\n__global__ void __launch_bounds__'
if 'thread_local int* spc_ids' not in s:
    pos=s.index(marker)
    s=s[:pos]+'''namespace { thread_local int* spc_ids = nullptr; thread_local float* spc_weights = nullptr; }
void FORWARD::setSpcDiagnostics(int* ids, float* weights) { spc_ids=ids; spc_weights=weights; }

'''+s[pos:]
    s=s.replace('float* __restrict__ out_color, const unsigned char* skip_tiles)', 'float* __restrict__ out_color, const unsigned char* skip_tiles, int* max_ids, float* max_weights)')
    s=s.replace('float C[CHANNELS] = { 0 }; // Auxiliary', 'float best_weight = 0.0f; int best_id = -1;\n\tfloat C[CHANNELS] = { 0 }; // Auxiliary')
    s=s.replace('\t\t\t// Eq. (3) from 3D Gaussian splatting paper.\n\t\t\tfor', '\t\t\tif (alpha*T > best_weight) { best_weight=alpha*T; best_id=collected_id[j]; }\n\t\t\t// Eq. (3) from 3D Gaussian splatting paper.\n\t\t\tfor',1)
    s=s.replace('\t\tfinal_T[pix_id] = T;', '\t\tif(max_ids) max_ids[pix_id]=best_id;\n\t\tif(max_weights) max_weights[pix_id]=best_weight;\n\t\tfinal_T[pix_id] = T;',1)
    s=s.replace('out_color, skip_tiles);', 'out_color, skip_tiles, spc_ids, spc_weights);')
    p.write_text(s,encoding='utf-8')
p=ROOT/'code/CMakeLists.txt'
s=p.read_text(encoding='utf-8')
if 'add_executable(spcgs_native' not in s:
    p.write_text(s+'\nadd_executable(spcgs_native examples/spcgs_native.cpp)\ntarget_link_libraries(spcgs_native gaussian_mapper)\n',encoding='utf-8')
print('Patched isolated core')
