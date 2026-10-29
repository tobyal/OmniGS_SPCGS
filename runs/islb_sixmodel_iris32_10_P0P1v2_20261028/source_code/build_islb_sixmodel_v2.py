"""Build an isolated adapter against the existing native OmniGS libraries."""
from pathlib import Path
import sys,shutil,shlex,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref
from spcgs.islb_sixmodel_data import RUN

def build():
    dest=RUN/'native';srcpath=ROOT/'code/examples/spcgs_native.cpp';src=srcpath.read_text()
    def change(old,new,count=1):
        nonlocal src
        assert src.count(old)==count,(old[:70],src.count(old),count)
        src=src.replace(old,new)
    helper='''
// Keep the established ERP rasterizer; sparse anchors can only optimize the
// shared semantic attributes and heads, never geometry, opacity or RGB SH.
std::shared_ptr<GaussianModel> semantic_only_model(std::shared_ptr<GaussianModel> m) {
    auto s=std::make_shared<GaussianModel>(m->max_sh_degree_);
    s->xyz_=m->xyz_.detach();s->features_dc_=m->features_dc_.detach();s->features_rest_=m->features_rest_.detach();
    s->scaling_=m->scaling_.detach();s->rotation_=m->rotation_.detach();s->opacity_=m->opacity_.detach();
    s->semantic_=m->semantic_;s->active_sh_degree_=m->active_sh_degree_;return s;
}
'''
    change('struct Heads {',helper+'\nstruct Heads {')
    change('T anchor_loss(Heads& h,T features,T xy,T labels,T class_scores,T* cosine,T* ce) {','T anchor_loss(Heads& h,T features,T xy,T labels,T class_scores,T* cosine,T* ce,bool ce_only=false) {')
    change('return clip_target_loss(h,sparse_features(features,xy),gt,scores,torch::ones({labels.numel()},GPU),false,cosine,ce);',
           'T lc,le;auto total=clip_target_loss(h,sparse_features(features,xy),gt,scores,torch::ones({labels.numel()},GPU),false,&lc,&le);if(cosine)*cosine=lc;if(ce)*ce=le;return ce_only?le:total;')
    change('&cosine,&ce);}', '&cosine,&ce,c.get("anchor_ce_only",false).asBool());}')
    change('auto r=render(m,v,H,W,HS,WS,true,false,false);T cosine,ce;auto loss=anchor_loss(',
           'auto r=render(semantic_only_model(m),v,H,W,HS,WS,true,false,false);T cosine,ce;auto loss=anchor_loss(')
    change('scores,&cosine,&ce);','scores,&cosine,&ce,c.get("anchor_ce_only",false).asBool());')
    change('auto ar=render(m,a.camera,H,W,HS,WS,true,false,false);','auto ar=render(semantic_only_model(m),a.camera,H,W,HS,WS,true,false,false);')
    change('la=anchor_loss(heads,ar.sem,a.xy,a.labels,class_scores,&anchor_cos,&anchor_ce);',
           'la=anchor_loss(heads,ar.sem,a.xy,a.labels,class_scores,&anchor_cos,&anchor_ce,c.get("anchor_ce_only",false).asBool());')
    change('ll=(semantics&&!heads.linear_ce)?local3d(', 'll=(semantics&&!heads.linear_ce&&c.get("local_weight",1.).asFloat()>0)?local3d(')
    change('row["anchor_view"]=a.camera["name"];', 'row["anchor_semantic_only"]=true;row["anchor_ce_only"]=c.get("anchor_ce_only",false);row["anchor_view"]=a.camera["name"];')
    change('std::vector<std::string> names={"semantic","decoder","bias","classifier","classifier_bias"};', '''auto geomgrads=torch::autograd::grad({la},{m->xyz_,m->scaling_,m->rotation_,m->opacity_,m->features_dc_},{},true,false,true);
                double geo_norm=0.;for(auto& g:geomgrads)if(g.defined())geo_norm+=g.abs().sum().item<double>();
                row["anchor_geometry_gradient_sum"]=geo_norm;TORCH_CHECK(geo_norm==0.,"Sparse anchor geometry must be detached");
                std::vector<std::string> names={"semantic","decoder","bias","classifier","classifier_bias"};''')
    probe='''
    if(action=="anchor_update_probe") {
        auto a=c["anchor_views"][0];int64_t n=a["anchor_count"].asInt64();
        std::vector<T> before;for(auto p:{&m->xyz_,&m->features_dc_,&m->features_rest_,&m->scaling_,&m->rotation_,&m->opacity_})before.push_back(p->detach().clone());
        auto sem_before=m->semantic_.detach().clone(),dec_before=heads.decoder.detach().clone(),text_before=heads.text.clone();
        auto ar=render(semantic_only_model(m),a,H,W,HS,WS,true,false,false);T lc,le;
        auto loss=anchor_loss(heads,ar.sem,raw(a["anchor_xy"].asString(),{n,2}),raw(a["anchor_labels"].asString(),{n},torch::kLong),raw(c["anchor_class_scores"].asString(),{K,K},torch::kFloat16).to(torch::kFloat32),&lc,&le,true)*c.get("anchor_weight",.05).asFloat();
        loss.backward();Json::Value result;result["loss"]=loss.item<double>();result["semantic_grad_norm"]=m->semantic_.grad().norm().item<double>();
        result["decoder_grad_norm"]=heads.decoder.grad().norm().item<double>();result["classifier_grad_norm"]=heads.classifier.grad().norm().item<double>();
        m->optimizer_->step();heads.optim->step();double maxgeo=0;int i=0;
        for(auto p:{&m->xyz_,&m->features_dc_,&m->features_rest_,&m->scaling_,&m->rotation_,&m->opacity_})maxgeo=std::max(maxgeo,(*p-before[i++]).abs().max().item<double>());
        result["geometry_RGB_max_change"]=maxgeo;result["semantic_max_change"]=(m->semantic_-sem_before).abs().max().item<double>();result["decoder_max_change"]=(heads.decoder-dec_before).abs().max().item<double>();
        result["text_unchanged"]=torch::equal(text_before,heads.text);result["optimizer_steps"]=1;result["anchor_CE_only"]=true;
        TORCH_CHECK(maxgeo==0. && result["semantic_max_change"].asDouble()>0 && result["decoder_max_change"].asDouble()>0 && result["text_unchanged"].asBool(),"Anchor-only update audit failed");
        writej(out/"anchor_update_validation.json",result);return 0;
    }
'''
    change('    if(action=="render") {torch::NoGradGuard g;',probe+'\n    if(action=="render") {torch::NoGradGuard g;')
    change('    save_now(out/"final",steps);return 0;', '    save_now(out/"final",steps);return 0;')
    change('if(step%c.get("save_every",500).asInt()==0||step==steps)save_now(out/("checkpoint_"+std::to_string(step)),step);',
        '''if(step%c.get("save_every",500).asInt()==0||step==steps)save_now(out/("checkpoint_"+std::to_string(step)),step);
        if(c.get("stop_after_step",0).asInt()==step && step<steps){save_now(out/("checkpoint_"+std::to_string(step)),step);Json::Value stop;stop["step"]=step;stop["reason"]="controlled resume preflight";writej(out/"controlled_stop.json",stop);return 0;}''')
    source=RUN/'source_code/spcgs_islb_sixmodel.cpp';source.write_text(src)
    libs=dest/'lib';libs.mkdir(exist_ok=True)
    for name in ['libgaussian_mapper.so','libcuda_rasterizer.so','libsimple_knn.so']:shutil.copy2(ROOT/'code/lib'/name,libs/name)
    builddir=ROOT/'code/build';flags=builddir/'CMakeFiles/spcgs_native_ISLB_anchor_v1.dir/flags.make'
    params={line.split(' = ',1)[0]:line.split(' = ',1)[1] for line in flags.read_text().splitlines() if ' = ' in line}
    obj=dest/'spcgs_islb_sixmodel.o';binary=dest/'spcgs_islb_sixmodel'
    compile_cmd=['/usr/bin/c++']+sum([shlex.split(params[k]) for k in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']],[])+['-c',str(source),'-o',str(obj)]
    link=shlex.split((builddir/'CMakeFiles/spcgs_native_ISLB_anchor_v1.dir/link.txt').read_text())
    link=[str(obj) if 'spcgs_native_ISLB_anchor_v1.dir/examples/spcgs_native.cpp.o' in x else x.replace(str(ROOT/'code/lib'),str(libs)) for x in link]
    link[link.index('-o')+1]=str(binary)
    write(dest/'build_commands.json',{'compile':compile_cmd,'link':link})
    with (dest/'build.log').open('w') as log:
        subprocess.run(compile_cmd,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=builddir)
        subprocess.run(link,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=builddir)
    write(dest/'build_audit.json',{'base_source':ref(srcpath),'adapter_source':ref(source),'binary':ref(binary),'OmniGS_libraries':[ref(p) for p in libs.glob('*.so')],
       'changes':['Anchor CE only option, shared CLIP-compatible head','Sparse-anchor geometry/RGB detached via a shallow tensor view of the native model','Anchor-only actual optimizer-step audit','Skip local regularization when weight0'],
       'renderer_modified':False,'Gaussian_optimizer_or_densification_modified':False,'historical_binary_overwritten':False})
    print('NATIVE_BUILD_READY',str(binary),flush=True)

if __name__=='__main__':build()
