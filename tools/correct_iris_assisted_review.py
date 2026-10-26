"""Focused corrections after actual RGB/overlay inspection; reuse SAM2 runner."""
import sys
import apply_iris_assisted_prompts as runner

P=[]
def add(cid,face,category,box,points,description,negative=()):
    P.append(dict(camera=cid,face=face,category=category,box=box,points=points,negative=negative,description=description))
add('1-32','down',12,[300,270,900,850],[[670,620],[450,570]],'恢复误选入脚架候选的可见地面',[[513,500],[400,300]])
add('1-32','down',255,[420,395,595,570],[[513,500],[467,517],[527,433],[560,537]],'按真实三条支腿重新提取脚架',[[450,450],[570,450],[500,560]])
add('1-56','right',27,[100,180,500,550],[[200,280],[410,400]],'恢复误选的背景白管为官方27',[[200,450]])
add('1-56','right',7,[0,350,450,730],[[200,450],[300,550]],'斜置控制盒的顶面与侧面，官方7',[[420,400],[410,670]])
add('1-56','right',9,[240,650,570,1000],[[390,800],[500,900]],'下方设备外壳可见顶面及侧面，官方9',[[470,650]])
add('1-56','right',255,[380,600,560,720],[[470,650]],'设备顶面独立纸张保持ignore',[[390,800]])
add('1-56','right',255,[0,600,220,780],[[60,680]],'控制盒左下纸张保持ignore')
add('1-45','left',27,[0,0,430,840],[[150,150],[100,400],[200,600]],'补齐前景白管可见上下两段',[[500,600],[500,300]])
add('1-45','left',16,[400,150,600,510],[[500,330]],'近景黑色立式泵，官方16')
runner.PROMPTS=P
if __name__=='__main__':runner.main()
