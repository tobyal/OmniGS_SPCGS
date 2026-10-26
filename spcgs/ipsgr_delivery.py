"""Package review assets; large checkpoints and all raw ISP/video data stay on the server."""
import tarfile
from pathlib import Path
from .common import read,write,ref
from .ipsgr_protocol import MODEL_NAMES
from .ipsgr_training import model_folder


def package(run):
    assert read(run/'experiment_manifest.json')['status']=='completed'
    selected=set()
    def add(folder,patterns):
        for pattern in patterns:
            selected.update(p for p in folder.glob(pattern) if p.is_file() and not p.is_symlink())
    add(run,['experiment_manifest.json','experiment_config.json'])
    add(run/'00_dataset',['*.json'])
    add(run/'08_paper_figures',['*.png','*.pdf','*.svg','*.json'])
    add(run/'09_paper_tables',['*.json','*.csv'])
    add(run/'07_industrial_analysis',['*.json','*.png','*.jpg'])
    add(run/'10_report',['report.html','report.md','viewer.html','viewer_manifest.json','completion_audit.json','final_model_validation.json','pre_isp_comparability.json','training_figure_evidence.json','resource_peaks.json','resource_samples.jsonl','runner_events.json','controller*.json'])
    add(run/'10_report/reproducibility',['*.json','*.txt','*.patch','final_code/**/*'])
    add(run/'06_evaluation/reference',['*.png','*.jpg'])
    for name in MODEL_NAMES:
        base=run/'06_evaluation'/name
        add(base,['*.json','*.png','query_maps/*.png','native/*_rgb.png','semantic_512x256/*_semantic.png','semantic_512x256/metrics.json'])
    for name in ['ISLB',*MODEL_NAMES]:
        base=model_folder(run,name)
        add(base,['*.json','losses.jsonl','native.log','representative_train_renders/**/*.png','representative_train_renders/*.json'])
    for name in ['P0','P1']:add(run/'01_geometry'/name,['*.json','*.f32','*.ply'])
    add(run/'01_geometry/Wan360',['generation_jobs.json','parallel_assignment*.json','parallel_completed.json','fresh_generation_audit.json'])
    add(run/'04_cross_view',['all_event_audit.json','saved_event_validation.json'])
    checkpoints={name:read(model_folder(run,name)/'outputs.json')['model'] for name in ['ISLB',*MODEL_NAMES]}
    index=run/'10_report/server_artifacts.json'
    write(index,{'server_run_directory':str(run),'checkpoints':checkpoints,'complete_raw_data_directories':{name:str(run/name) for name in ['00_dataset','01_geometry','02_semantic_teacher','03_layout_bootstrap','04_cross_view','05_models','06_evaluation']},'review_archive_scope':'Reports, figures, tables, final test display images, training logs, protocol and compact geometry evidence. Full checkpoint archives, videos, floating-point render buffers and every ISP event remain saved in the server run directory.'})
    selected.add(index)
    manifest=run/'10_report/review_package_manifest.json'
    write(manifest,{'status':'completed','server_run_directory':str(run),'files':[{'relative_path':str(p.relative_to(run)),**ref(p)} for p in sorted(selected)]})
    selected.add(manifest)
    archive=run/'10_report/review_package.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(selected):tar.add(path,arcname=str(path.relative_to(run)),recursive=False)
    result={'archive':ref(archive),'files':len(selected),'raw_checkpoints_retained_on_server':True}
    write(run/'10_report/review_package.json',result)
    return result
