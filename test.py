from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

df0=pd.read_csv("data/new/fujitaRP1.csv",index_col=1)
df2=pd.read_csv("data/new/koyamaRP1.csv",index_col=1)
df3=pd.read_csv("data/new/koyanagiRP1.csv",index_col=1)
df4=pd.read_csv("data/new/nimuraRP1.csv",index_col=1)
df5=pd.read_csv("data/new/suzukiRP1.csv",index_col=1)
df6_c=pd.read_csv("data/new/64058436RP1.csv",index_col=1)
df7_c=pd.read_csv("data/new/79122114RP1.csv",index_col=1)
df8_c=pd.read_csv("data/new/52193348RP1.csv",index_col=1)
df9_c=pd.read_csv("data/new/19099510RP1.csv",index_col=1)
df10_c=pd.read_csv("data/new/19110622RP1.csv",index_col=1)
df11_c=pd.read_csv("data/new/19630516RP1.csv",index_col=1)
df12_c=pd.read_csv("data/new/82304270RP1.csv",index_col=1)
df13_c=pd.read_csv("data/new/15662472RP1.csv",index_col=1)
df14_c=pd.read_csv("data/new/19600594RP1.csv",index_col=1)
df15_c=pd.read_csv("data/new/73209319RP1.csv",index_col=1)
df16_c=pd.read_csv("data/new/18611860R_0.csv",index_col=1) # ここ以降手首の角度あります。
df17_c=pd.read_csv("data/new/19619780R_0.csv",index_col=1)
df18_c=pd.read_csv("data/new/19636280R0.csv",index_col=1)
df19_c=pd.read_csv("data/new/19639492R_0.csv",index_col=1)
df20_c=pd.read_csv("data/new/67090028R_0.csv",index_col=1)
df21_c=pd.read_csv("data/new/81160006R_0.csv",index_col=1)
df22=pd.read_csv("data/masaru_0.csv",index_col=1)
df23=pd.read_csv("data/etukoRP1.csv",index_col=1)
df24=pd.read_csv("data/turu22.csv",index_col=1)
#df25=pd.read_csv("data/turu23.csv",index_col=1)
df26=pd.read_csv("data/turu20.csv",index_col=1)
df27=pd.read_csv("data/turu19.csv",index_col=1)
df28=pd.read_csv("data/turu18.csv",index_col=1)
#df29=pd.read_csv("data/turu15.csv",index_col=1)
#df30=pd.read_csv("data/turu14_0.csv",index_col=1)
df31=pd.read_csv("data/turu11.csv",index_col=1)
df32=pd.read_csv("data/turu10.csv",index_col=1)
df33=pd.read_csv("data/turu9_0.csv",index_col=1)
#df34=pd.read_csv("data/turu1.csv",index_col=1)
df35_c=pd.read_csv("data/18695075.csv",index_col=1)
df36_c=pd.read_csv("data/19025608.csv",index_col=1)
df37_c=pd.read_csv("data/19633531.csv",index_col=1)
df38_c=pd.read_csv("data/19638894.csv",index_col=1)
df39_c=pd.read_csv("data/88621754.csv",index_col=1)
df40_c=pd.read_csv("data/86140331_c.csv",index_col=1)
df41_c=pd.read_csv("data/19639054_c.csv",index_col=1)
df42_c=pd.read_csv("data/19629494_c.csv",index_col=1)
df43_c=pd.read_csv("data/19169481_c.csv",index_col=1)
df44_c=pd.read_csv("data/17167931_c.csv",index_col=1)
df45_c=pd.read_csv("data/16651981_c.csv",index_col=1)
df46_c=pd.read_csv("data/16639522_c.csv",index_col=1)
df47_c=pd.read_csv("data/15620160_c.csv",index_col=1)
df48=pd.read_csv("data/74123588.csv",index_col=1)
df49=pd.read_csv("data/73043426.csv",index_col=1)
df50=pd.read_csv("data/19653527.csv",index_col=1)
df51=pd.read_csv("data/19652314.csv",index_col=1)
df52=pd.read_csv("data/19651592.csv",index_col=1)
df53=pd.read_csv("data/19648231.csv",index_col=1)
df54=pd.read_csv("data/19630892.csv",index_col=1)
df55=pd.read_csv("data/19140706.csv",index_col=1)
df56=pd.read_csv("data/18693077.csv",index_col=1)
df57=pd.read_csv("data/18642391.csv",index_col=1)
df58_c=pd.read_csv("data/18684050_c.csv",index_col=1)
df59_c=pd.read_csv("data/19627664_c.csv",index_col=1)
df60_c=pd.read_csv("data/19639606_c.csv",index_col=1)
df61_c=pd.read_csv("data/19641229_c.csv",index_col=1)
df62_c=pd.read_csv("data/19648874_c.csv",index_col=1)
df63_c=pd.read_csv("data/19650334_c.csv",index_col=1)
df64_c=pd.read_csv("data/19662018_c.csv",index_col=1)
df65_c=pd.read_csv("data/19680864_c.csv",index_col=1)
df66_c=pd.read_csv("data/86141143_c.csv",index_col=1)
df67_c=pd.read_csv("data/87175907_c.csv",index_col=1)
df68=pd.read_csv("data/15012763.csv",index_col=1)
df69=pd.read_csv("data/16605773.csv",index_col=1)
df70=pd.read_csv("data/19632168.csv",index_col=1)
df71=pd.read_csv("data/19674871.csv",index_col=1)
df72=pd.read_csv("data/19684263.csv",index_col=1)
df73=pd.read_csv("data/61176723.csv",index_col=1)
df74=pd.read_csv("data/79303810.csv",index_col=1)
df75=pd.read_csv("data/85226517.csv",index_col=1)
df76_c=pd.read_csv("data/20204/5299463_c.csv",index_col=1)
df77_c=pd.read_csv("data/20204/19642763_c.csv",index_col=1)
df77_c=pd.read_csv("data/20204/19679037_c.csv",index_col=1)
df79_c=pd.read_csv("data/20204/19684183_c.csv",index_col=1)
df80_c=pd.read_csv("data/20204/20039931_c.csv",index_col=1)
df81_c=pd.read_csv("data/20204/20050392_c.csv",index_col=1)
df82=pd.read_csv("data/20204/19628315.csv",index_col=1)
df83=pd.read_csv("data/20204/19652261.csv",index_col=1)
df84=pd.read_csv("data/20204/19698855.csv",index_col=1)
df85=pd.read_csv("data/20204/19702573.csv",index_col=1)
df86=pd.read_csv("data/20204/20602448.csv",index_col=1)
df87=pd.read_csv("data/20204/20606649.csv",index_col=1)
df88=pd.read_csv("data/20204/76066920.csv",index_col=1)
df89=pd.read_csv("data/20204/85293695.csv",index_col=1)

# 左右２回ずつのデータ
data1=pd.read_csv("datasRL/52193348L1_c.csv",index_col=1)
data2=pd.read_csv("datasRL/52193348L2_c.csv",index_col=1)
data3=pd.read_csv("datasRL/52193348R1_c.csv",index_col=1)
data4=pd.read_csv("datasRL/52193348R2_c.csv",index_col=1)
data5=pd.read_csv("datasRL/5299463L1_c.csv",index_col=1)
data6=pd.read_csv("datasRL/5299463L2_c.csv",index_col=1)
data7=pd.read_csv("datasRL/5299463R1_c.csv",index_col=1)
data8=pd.read_csv("datasRL/5299463R2_c.csv",index_col=1)
data9=pd.read_csv("datasRL/64058436L1_c.csv",index_col=1)
data10=pd.read_csv("datasRL/64058436L2_c.csv",index_col=1)
data11=pd.read_csv("datasRL/64058436R1_c.csv",index_col=1)
data12=pd.read_csv("datasRL/64058436R2_c.csv",index_col=1)
data13=pd.read_csv("datasRL/79122114L1_c.csv",index_col=1)
data14=pd.read_csv("datasRL/79122114L2_c.csv",index_col=1)
data15=pd.read_csv("datasRL/79122114R1_c.csv",index_col=1)
data16=pd.read_csv("datasRL/79122114R2_c.csv",index_col=1)
data17=pd.read_csv("datasRL/20190606/19099510L1_c.csv",index_col=1)
data18=pd.read_csv("datasRL/20190606/19099510L2_c.csv",index_col=1)
data19=pd.read_csv("datasRL/20190606/19099510R1_c.csv",index_col=1)
data20=pd.read_csv("datasRL/20190606/19099510R2_c.csv",index_col=1)
data21=pd.read_csv("datasRL/20190606/19110622L1_c.csv",index_col=1)
data22=pd.read_csv("datasRL/20190606/19110622L2_c.csv",index_col=1)
data23=pd.read_csv("datasRL/20190606/19110622R1_c.csv",index_col=1)
data24=pd.read_csv("datasRL/20190606/19110622R2_c.csv",index_col=1)
data25=pd.read_csv("datasRL/20190606/19630516L1_c.csv",index_col=1)
data26=pd.read_csv("datasRL/20190606/19630516L2_c.csv",index_col=1)
data27=pd.read_csv("datasRL/20190606/19630516R1_c.csv",index_col=1)
data28=pd.read_csv("datasRL/20190606/19630516R2_c.csv",index_col=1)
data29=pd.read_csv("datasRL/20190624/15662472L1_c.csv",index_col=1)
data30=pd.read_csv("datasRL/20190624/15662472L2_c.csv",index_col=1)
data31=pd.read_csv("datasRL/20190624/15662472R1_c.csv",index_col=1)
data32=pd.read_csv("datasRL/20190624/15662472R2_c.csv",index_col=1)
data33=pd.read_csv("datasRL/20190624/19600594L1_c.csv",index_col=1)
data34=pd.read_csv("datasRL/20190624/19600594L2_c.csv",index_col=1)
data35=pd.read_csv("datasRL/20190624/19600594R1_c.csv",index_col=1)
data36=pd.read_csv("datasRL/20190624/19600594R2_c.csv",index_col=1)
# data37=pd.read_csv("datasRL/20190624/79122114L1_c.csv",index_col=1)
# data38=pd.read_csv("datasRL/20190624/79122114L2_c.csv",index_col=1)
# data39=pd.read_csv("datasRL/20190624/79122114R1_c.csv",index_col=1)
# data40=pd.read_csv("datasRL/20190624/79122114R2_c.csv",index_col=1)
data41=pd.read_csv("datasRL/20190624/73209319L1_c.csv",index_col=1)
data42=pd.read_csv("datasRL/20190624/73209319L2_c.csv",index_col=1)
data43=pd.read_csv("datasRL/20190624/73209319R1_c.csv",index_col=1)
data44=pd.read_csv("datasRL/20190624/73209319R2_c.csv",index_col=1)
data45=pd.read_csv("datasRL/20190624/82304270L1_c.csv",index_col=1)
data46=pd.read_csv("datasRL/20190624/82304270L2_c.csv",index_col=1)
data47=pd.read_csv("datasRL/20190624/82304270R1_c.csv",index_col=1)
data48=pd.read_csv("datasRL/20190624/82304270R2_c.csv",index_col=1)
data49=pd.read_csv("datasRL/20190704/18611860L1_c.csv",index_col=1)
data50=pd.read_csv("datasRL/20190704/18611860L2_c.csv",index_col=1)
data51=pd.read_csv("datasRL/20190704/18611860R1_c.csv",index_col=1)
data52=pd.read_csv("datasRL/20190704/18611860R2_c.csv",index_col=1)
data53=pd.read_csv("datasRL/20190704/19619780L1_c.csv",index_col=1)
data54=pd.read_csv("datasRL/20190704/19619780L2_c.csv",index_col=1)
data55=pd.read_csv("datasRL/20190704/19619780R1_c.csv",index_col=1)
data56=pd.read_csv("datasRL/20190704/19619780R2_c.csv",index_col=1)
data57=pd.read_csv("datasRL/20190704/19636280L1_c.csv",index_col=1)
data58=pd.read_csv("datasRL/20190704/19636280L2_c.csv",index_col=1)
data59=pd.read_csv("datasRL/20190704/19636280R1_c.csv",index_col=1)
data60=pd.read_csv("datasRL/20190704/19636280R2_c.csv",index_col=1)
data61=pd.read_csv("datasRL/20190704/19639492L1_c.csv",index_col=1)
data62=pd.read_csv("datasRL/20190704/19639492L2_c.csv",index_col=1)
data63=pd.read_csv("datasRL/20190704/19639492R1_c.csv",index_col=1)
data64=pd.read_csv("datasRL/20190704/19639492R2_c.csv",index_col=1)
data65=pd.read_csv("datasRL/20190704/67090028L1_c.csv",index_col=1)
data66=pd.read_csv("datasRL/20190704/67090028L2_c.csv",index_col=1)
data67=pd.read_csv("datasRL/20190704/67090028R1_c.csv",index_col=1)
data68=pd.read_csv("datasRL/20190704/67090028R2_c.csv",index_col=1)
data69=pd.read_csv("datasRL/20190704/81160006L1_c.csv",index_col=1)
data70=pd.read_csv("datasRL/20190704/81160006L2_c.csv",index_col=1)
data71=pd.read_csv("datasRL/20190704/81160006R1_c.csv",index_col=1)
data72=pd.read_csv("datasRL/20190704/81160006R2_c.csv",index_col=1)
data73=pd.read_csv("datasRL/20190725/18695075L1_c.csv",index_col=1)
data74=pd.read_csv("datasRL/20190725/18695075L2_c.csv",index_col=1)
data75=pd.read_csv("datasRL/20190725/18695075R1_c.csv",index_col=1)
data76=pd.read_csv("datasRL/20190725/18695075R2_c.csv",index_col=1)
data77=pd.read_csv("datasRL/20190725/19025608L1_c.csv",index_col=1)
data78=pd.read_csv("datasRL/20190725/19025608L2_c.csv",index_col=1)
data79=pd.read_csv("datasRL/20190725/19025608R1_c.csv",index_col=1)
data80=pd.read_csv("datasRL/20190725/19025608R2_c.csv",index_col=1)
data81=pd.read_csv("datasRL/20190725/19633531L1_c.csv",index_col=1)
data82=pd.read_csv("datasRL/20190725/19633531L2_c.csv",index_col=1)
data83=pd.read_csv("datasRL/20190725/19633531R1_c.csv",index_col=1)
data84=pd.read_csv("datasRL/20190725/19633531R2_c.csv",index_col=1)
data85=pd.read_csv("datasRL/20190725/19638894L1_c.csv",index_col=1)
data86=pd.read_csv("datasRL/20190725/19638894L2_c.csv",index_col=1)
data87=pd.read_csv("datasRL/20190725/19638894R1_c.csv",index_col=1)
data88=pd.read_csv("datasRL/20190725/19638894R2_c.csv",index_col=1)
data89=pd.read_csv("datasRL/20190725/88621754L1_c.csv",index_col=1)
data90=pd.read_csv("datasRL/20190725/88621754L2_c.csv",index_col=1)
data91=pd.read_csv("datasRL/20190725/88621754R1_c.csv",index_col=1)
data92=pd.read_csv("datasRL/20190725/88621754R2_c.csv",index_col=1)
data93=pd.read_csv("datasRL/20190819/16651981L1_c.csv",index_col=1)
data94=pd.read_csv("datasRL/20190819/16651981L2_c.csv",index_col=1)
data95=pd.read_csv("datasRL/20190819/16651981R1_c.csv",index_col=1)
data96=pd.read_csv("datasRL/20190819/16651981R2_c.csv",index_col=1)
data97=pd.read_csv("datasRL/20190819/18642391L1.csv",index_col=1)
data98=pd.read_csv("datasRL/20190819/18642391L2.csv",index_col=1)
data99=pd.read_csv("datasRL/20190819/18642391R1.csv",index_col=1)
data100=pd.read_csv("datasRL/20190819/18642391R2.csv",index_col=1)
data101=pd.read_csv("datasRL/20190819/19140706L1.csv",index_col=1)
data102=pd.read_csv("datasRL/20190819/19140706L2.csv",index_col=1)
data103=pd.read_csv("datasRL/20190819/19140706R1.csv",index_col=1)
data104=pd.read_csv("datasRL/20190819/19140706R2.csv",index_col=1)
data105=pd.read_csv("datasRL/20190819/19629494L1_c.csv",index_col=1)
data106=pd.read_csv("datasRL/20190819/19629494L2_c.csv",index_col=1)
data107=pd.read_csv("datasRL/20190819/19629494R1_c.csv",index_col=1)
data108=pd.read_csv("datasRL/20190819/19629494R2_c.csv",index_col=1)
data109=pd.read_csv("datasRL/20190819/19630892L1.csv",index_col=1)
data110=pd.read_csv("datasRL/20190819/19630892L2.csv",index_col=1)
data111=pd.read_csv("datasRL/20190819/19630892R1.csv",index_col=1)
data112=pd.read_csv("datasRL/20190819/19630892R2.csv",index_col=1)
data113=pd.read_csv("datasRL/20190819/19651592L1.csv",index_col=1)
data114=pd.read_csv("datasRL/20190819/19651592L2.csv",index_col=1)
data115=pd.read_csv("datasRL/20190819/19651592R1.csv",index_col=1)
data116=pd.read_csv("datasRL/20190819/19651592R2.csv",index_col=1)
data117=pd.read_csv("datasRL/20190819/19652314L1.csv",index_col=1)
data118=pd.read_csv("datasRL/20190819/19652314L2.csv",index_col=1)
data119=pd.read_csv("datasRL/20190819/19652314R1.csv",index_col=1)
data120=pd.read_csv("datasRL/20190819/19652314R2.csv",index_col=1)
data121=pd.read_csv("datasRL/20190920/15620160L1_c.csv",index_col=1)
data122=pd.read_csv("datasRL/20190920/15620160L2_c.csv",index_col=1)
data123=pd.read_csv("datasRL/20190920/15620160R1_c.csv",index_col=1)
data124=pd.read_csv("datasRL/20190920/15620160R2_c.csv",index_col=1)
data125=pd.read_csv("datasRL/20190920/16639522L1_c.csv",index_col=1)
data126=pd.read_csv("datasRL/20190920/16639522L2_c.csv",index_col=1)
data127=pd.read_csv("datasRL/20190920/16639522R1_c.csv",index_col=1)
data128=pd.read_csv("datasRL/20190920/16639522R2_c.csv",index_col=1)
data129=pd.read_csv("datasRL/20190920/17167931L1_c.csv",index_col=1)
data130=pd.read_csv("datasRL/20190920/17167931L2_c.csv",index_col=1)
data131=pd.read_csv("datasRL/20190920/17167931R1_c.csv",index_col=1)
data132=pd.read_csv("datasRL/20190920/17167931R2_c.csv",index_col=1)
data133=pd.read_csv("datasRL/20190920/18693077L1.csv",index_col=1)
data134=pd.read_csv("datasRL/20190920/18693077L2.csv",index_col=1)
data135=pd.read_csv("datasRL/20190920/18693077R1.csv",index_col=1)
data136=pd.read_csv("datasRL/20190920/18693077R2.csv",index_col=1)
data137=pd.read_csv("datasRL/20190920/19169481L1_c.csv",index_col=1)
data138=pd.read_csv("datasRL/20190920/19169481L2_c.csv",index_col=1)
data139=pd.read_csv("datasRL/20190920/19169481R1_c.csv",index_col=1)
data140=pd.read_csv("datasRL/20190920/19169481R2_c.csv",index_col=1)
data141=pd.read_csv("datasRL/20190920/19639054L1_c.csv",index_col=1)
data142=pd.read_csv("datasRL/20190920/19639054L2_c.csv",index_col=1)
data143=pd.read_csv("datasRL/20190920/19639054R1_c.csv",index_col=1)
data144=pd.read_csv("datasRL/20190920/19639054R2_c.csv",index_col=1)
data145=pd.read_csv("datasRL/20190920/19648231L1.csv",index_col=1)
data146=pd.read_csv("datasRL/20190920/19648231L2.csv",index_col=1)
data147=pd.read_csv("datasRL/20190920/19648231R1.csv",index_col=1)
data148=pd.read_csv("datasRL/20190920/19648231R2.csv",index_col=1)
data149=pd.read_csv("datasRL/20190920/19653527L1.csv",index_col=1)
data150=pd.read_csv("datasRL/20190920/19653527L2.csv",index_col=1)
data151=pd.read_csv("datasRL/20190920/19653527R1.csv",index_col=1)
data152=pd.read_csv("datasRL/20190920/19653527R2.csv",index_col=1)
data153=pd.read_csv("datasRL/20190920/73043426L1.csv",index_col=1)
data154=pd.read_csv("datasRL/20190920/73043426L2.csv",index_col=1)
data155=pd.read_csv("datasRL/20190920/73043426R1.csv",index_col=1)
data156=pd.read_csv("datasRL/20190920/73043426R2.csv",index_col=1)
data157=pd.read_csv("datasRL/20190920/74123588L1.csv",index_col=1)
data158=pd.read_csv("datasRL/20190920/74123588L2.csv",index_col=1)
data159=pd.read_csv("datasRL/20190920/74123588R1.csv",index_col=1)
data160=pd.read_csv("datasRL/20190920/74123588R2.csv",index_col=1)
data161=pd.read_csv("datasRL/20190920/86140331L1_c.csv",index_col=1)
data162=pd.read_csv("datasRL/20190920/86140331L2_c.csv",index_col=1)
data163=pd.read_csv("datasRL/20190920/86140331R1_c.csv",index_col=1)
data164=pd.read_csv("datasRL/20190920/86140331R2_c.csv",index_col=1)
data165=pd.read_csv("datasRL/20191108/18684050L1_c.csv",index_col=1)
data166=pd.read_csv("datasRL/20191108/18684050L2_c.csv",index_col=1)
data167=pd.read_csv("datasRL/20191108/18684050R1_c.csv",index_col=1)
data168=pd.read_csv("datasRL/20191108/18684050R2_c.csv",index_col=1)
data160=pd.read_csv("datasRL/20191108/19627664L1_c.csv",index_col=1)
data170=pd.read_csv("datasRL/20191108/19627664L2_c.csv",index_col=1)
data171=pd.read_csv("datasRL/20191108/19627664R1_c.csv",index_col=1)
data172=pd.read_csv("datasRL/20191108/19627664R2_c.csv",index_col=1)
data173=pd.read_csv("datasRL/20191108/19632168L1.csv",index_col=1)
data174=pd.read_csv("datasRL/20191108/19632168L2.csv",index_col=1)
data175=pd.read_csv("datasRL/20191108/19632168R1.csv",index_col=1)
data176=pd.read_csv("datasRL/20191108/19632168R2.csv",index_col=1)
data177=pd.read_csv("datasRL/20191108/19648874L1_c.csv",index_col=1)
data178=pd.read_csv("datasRL/20191108/19648874L2_c.csv",index_col=1)
data179=pd.read_csv("datasRL/20191108/19648874R1_c.csv",index_col=1)
data180=pd.read_csv("datasRL/20191108/19648874R2_c.csv",index_col=1)
data181=pd.read_csv("datasRL/20191108/19650334L1_c.csv",index_col=1)
data182=pd.read_csv("datasRL/20191108/19650334L2_c.csv",index_col=1)
data183=pd.read_csv("datasRL/20191108/19650334R1_c.csv",index_col=1)
data184=pd.read_csv("datasRL/20191108/19650334R2_c.csv",index_col=1)
data185=pd.read_csv("datasRL/20191108/79303810L1.csv",index_col=1)
data186=pd.read_csv("datasRL/20191108/79303810L2.csv",index_col=1)
data187=pd.read_csv("datasRL/20191108/79303810R1.csv",index_col=1)
data188=pd.read_csv("datasRL/20191108/79303810R2.csv",index_col=1)
data189=pd.read_csv("datasRL/20191108/85226517L1.csv",index_col=1)
data190=pd.read_csv("datasRL/20191108/85226517L2.csv",index_col=1)
data191=pd.read_csv("datasRL/20191108/85226517R1.csv",index_col=1)
data192=pd.read_csv("datasRL/20191108/85226517R2.csv",index_col=1)
data193=pd.read_csv("datasRL/20191108/87175907L1_c.csv",index_col=1)
data194=pd.read_csv("datasRL/20191108/87175907L2_c.csv",index_col=1)
data195=pd.read_csv("datasRL/20191108/87175907R1_c.csv",index_col=1)
data196=pd.read_csv("datasRL/20191108/87175907R2_c.csv",index_col=1)
data197=pd.read_csv("datasRL/20191212/15012763L1.csv",index_col=1)
data198=pd.read_csv("datasRL/20191212/15012763L2.csv",index_col=1)
data199=pd.read_csv("datasRL/20191212/15012763R1.csv",index_col=1)
data200=pd.read_csv("datasRL/20191212/15012763R2.csv",index_col=1)
data201=pd.read_csv("datasRL/20191212/16605773L1.csv",index_col=1)
data202=pd.read_csv("datasRL/20191212/16605773L2.csv",index_col=1)
data203=pd.read_csv("datasRL/20191212/16605773R1.csv",index_col=1)
data204=pd.read_csv("datasRL/20191212/16605773R2.csv",index_col=1)
data205=pd.read_csv("datasRL/20191212/19639606L1_c.csv",index_col=1)
data206=pd.read_csv("datasRL/20191212/19639606L2_c.csv",index_col=1)
data207=pd.read_csv("datasRL/20191212/19639606R1_c.csv",index_col=1)
data208=pd.read_csv("datasRL/20191212/19639606R2_c.csv",index_col=1)
data209=pd.read_csv("datasRL/20191212/19641229L1_c.csv",index_col=1)
data210=pd.read_csv("datasRL/20191212/19641229L2_c.csv",index_col=1)
data211=pd.read_csv("datasRL/20191212/19641229R1_c.csv",index_col=1)
data212=pd.read_csv("datasRL/20191212/19641229R2_c.csv",index_col=1)
data213=pd.read_csv("datasRL/20191212/19662018L1_c.csv",index_col=1)
data214=pd.read_csv("datasRL/20191212/19662018L2_c.csv",index_col=1)
data215=pd.read_csv("datasRL/20191212/19662018R1_c.csv",index_col=1)
data216=pd.read_csv("datasRL/20191212/19662018R2_c.csv",index_col=1)
data217=pd.read_csv("datasRL/20191212/19674871L1.csv",index_col=1)
data218=pd.read_csv("datasRL/20191212/19674871L2.csv",index_col=1)
data219=pd.read_csv("datasRL/20191212/19674871R1.csv",index_col=1)
data220=pd.read_csv("datasRL/20191212/19674871R2.csv",index_col=1)
data221=pd.read_csv("datasRL/20191212/19680864L1_c.csv",index_col=1)
data222=pd.read_csv("datasRL/20191212/19680864L2_c.csv",index_col=1)
data223=pd.read_csv("datasRL/20191212/19680864R1_c.csv",index_col=1)
data224=pd.read_csv("datasRL/20191212/19680864R2_c.csv",index_col=1)
data225=pd.read_csv("datasRL/20191212/19684263L1.csv",index_col=1)
data226=pd.read_csv("datasRL/20191212/19684263L2.csv",index_col=1)
data227=pd.read_csv("datasRL/20191212/19684263R1.csv",index_col=1)
data228=pd.read_csv("datasRL/20191212/19684263R2.csv",index_col=1)
data229=pd.read_csv("datasRL/20191212/61176723L1.csv",index_col=1)
data230=pd.read_csv("datasRL/20191212/61176723L2.csv",index_col=1)
data231=pd.read_csv("datasRL/20191212/61176723R1.csv",index_col=1)
data232=pd.read_csv("datasRL/20191212/61176723R2.csv",index_col=1)
data233=pd.read_csv("datasRL/20191212/86141143L1_c.csv",index_col=1)
data234=pd.read_csv("datasRL/20191212/86141143L2_c.csv",index_col=1)
data235=pd.read_csv("datasRL/20191212/86141143R1_c.csv",index_col=1)
data236=pd.read_csv("datasRL/20191212/86141143R2_c.csv",index_col=1)
data237=pd.read_csv("datasRL/20200107/16629365L1_c.csv",index_col=1)
data238=pd.read_csv("datasRL/20200107/16629365L2_c.csv",index_col=1)
data239=pd.read_csv("datasRL/20200107/16629365R1_c.csv",index_col=1)
data240=pd.read_csv("datasRL/20200107/16629365R2_c.csv",index_col=1)
data241=pd.read_csv("datasRL/20200107/19284793L1_c.csv",index_col=1)
data242=pd.read_csv("datasRL/20200107/19284793L2_c.csv",index_col=1)
data243=pd.read_csv("datasRL/20200107/19284793R1_c.csv",index_col=1)
data244=pd.read_csv("datasRL/20200107/19284793R2_c.csv",index_col=1)
data245=pd.read_csv("datasRL/20200107/19643101L1.csv",index_col=1)
data246=pd.read_csv("datasRL/20200107/19643101L2.csv",index_col=1)
data247=pd.read_csv("datasRL/20200107/19643101R1.csv",index_col=1)
data248=pd.read_csv("datasRL/20200107/19643101R2.csv",index_col=1)
data249=pd.read_csv("datasRL/20200107/19661019L1_c.csv",index_col=1)
data250=pd.read_csv("datasRL/20200107/19661019L2_c.csv",index_col=1)
data251=pd.read_csv("datasRL/20200107/19661019R1_c.csv",index_col=1)
data252=pd.read_csv("datasRL/20200107/19661019R2_c.csv",index_col=1)
data253=pd.read_csv("datasRL/20200107/19661251L1.csv",index_col=1)
data254=pd.read_csv("datasRL/20200107/19661251L2.csv",index_col=1)
data255=pd.read_csv("datasRL/20200107/19661251R1.csv",index_col=1)
data256=pd.read_csv("datasRL/20200107/19661251R2.csv",index_col=1)
data257=pd.read_csv("datasRL/20200107/79056150L1_c.csv",index_col=1)
data258=pd.read_csv("datasRL/20200107/79056150L2_c.csv",index_col=1)
data259=pd.read_csv("datasRL/20200107/79056150R1_c.csv",index_col=1)
data260=pd.read_csv("datasRL/20200107/79056150R2_c.csv",index_col=1)
data261=pd.read_csv("datasRL/20200107/87363277L1_c.csv",index_col=1)
data262=pd.read_csv("datasRL/20200107/87363277L2_c.csv",index_col=1)
data263=pd.read_csv("datasRL/20200107/87363277R1_c.csv",index_col=1)
data264=pd.read_csv("datasRL/20200107/87363277R2_c.csv",index_col=1)
data265=pd.read_csv("datasRL/20200408/19628315L1.csv",index_col=1)
data266=pd.read_csv("datasRL/20200408/19628315L2.csv",index_col=1)
data267=pd.read_csv("datasRL/20200408/19628315R1.csv",index_col=1)
data268=pd.read_csv("datasRL/20200408/19628315R2.csv",index_col=1)
data269=pd.read_csv("datasRL/20200408/19642763L1_c.csv",index_col=1)
data270=pd.read_csv("datasRL/20200408/19642763L2_c.csv",index_col=1)
data271=pd.read_csv("datasRL/20200408/19642763R1_c.csv",index_col=1)
data272=pd.read_csv("datasRL/20200408/19642763R2_c.csv",index_col=1)
data273=pd.read_csv("datasRL/20200408/19652261L1.csv",index_col=1)
data274=pd.read_csv("datasRL/20200408/19652261L2.csv",index_col=1)
data275=pd.read_csv("datasRL/20200408/19652261R1.csv",index_col=1)
data276=pd.read_csv("datasRL/20200408/19652261R2.csv",index_col=1)
data277=pd.read_csv("datasRL/20200408/19679037L1_c.csv",index_col=1)
data278=pd.read_csv("datasRL/20200408/19679037L2_c.csv",index_col=1)
data279=pd.read_csv("datasRL/20200408/19679037R1_c.csv",index_col=1)
data280=pd.read_csv("datasRL/20200408/19679037R2_c.csv",index_col=1)
data281=pd.read_csv("datasRL/20200408/19684183L1_c.csv",index_col=1)
data282=pd.read_csv("datasRL/20200408/19684183L2_c.csv",index_col=1)
data283=pd.read_csv("datasRL/20200408/19684183R1_c.csv",index_col=1)
data284=pd.read_csv("datasRL/20200408/19684183R2_c.csv",index_col=1)
data285=pd.read_csv("datasRL/20200408/19698855L1.csv",index_col=1)
data286=pd.read_csv("datasRL/20200408/19698855L2.csv",index_col=1)
data287=pd.read_csv("datasRL/20200408/19698855R1.csv",index_col=1)
data288=pd.read_csv("datasRL/20200408/19698855R2.csv",index_col=1)
data289=pd.read_csv("datasRL/20200408/19702573L1.csv",index_col=1)
data290=pd.read_csv("datasRL/20200408/19702573L2.csv",index_col=1)
data291=pd.read_csv("datasRL/20200408/19702573R1.csv",index_col=1)
data292=pd.read_csv("datasRL/20200408/19702573R2.csv",index_col=1)
data293=pd.read_csv("datasRL/20200408/20039931L1_c.csv",index_col=1)
data294=pd.read_csv("datasRL/20200408/20039931L2_c.csv",index_col=1)
data295=pd.read_csv("datasRL/20200408/20039931R1_c.csv",index_col=1)
data296=pd.read_csv("datasRL/20200408/20039931R2_c.csv",index_col=1)
data297=pd.read_csv("datasRL/20200408/20050392L1_c.csv",index_col=1)
data298=pd.read_csv("datasRL/20200408/20050392L2_c.csv",index_col=1)
data299=pd.read_csv("datasRL/20200408/20050392R1_c.csv",index_col=1)
data300=pd.read_csv("datasRL/20200408/20050392R2_c.csv",index_col=1)
data301=pd.read_csv("datasRL/20200408/20602448L1.csv",index_col=1)
data302=pd.read_csv("datasRL/20200408/20602448L2.csv",index_col=1)
data303=pd.read_csv("datasRL/20200408/20602448R1.csv",index_col=1)
data304=pd.read_csv("datasRL/20200408/20602448R2.csv",index_col=1)
data305=pd.read_csv("datasRL/20200408/20606649L1.csv",index_col=1)
data306=pd.read_csv("datasRL/20200408/20606649L2.csv",index_col=1)
data307=pd.read_csv("datasRL/20200408/20606649R1.csv",index_col=1)
data308=pd.read_csv("datasRL/20200408/20606649R2.csv",index_col=1)
data309=pd.read_csv("datasRL/20200408/76066920L1.csv",index_col=1)
data310=pd.read_csv("datasRL/20200408/76066920L2.csv",index_col=1)
data311=pd.read_csv("datasRL/20200408/76066920R1.csv",index_col=1)
data312=pd.read_csv("datasRL/20200408/76066920R2.csv",index_col=1)
data313=pd.read_csv("datasRL/20200408/85293695L1.csv",index_col=1)
data314=pd.read_csv("datasRL/20200408/85293695L2.csv",index_col=1)
data315=pd.read_csv("datasRL/20200408/85293695R1.csv",index_col=1)
data316=pd.read_csv("datasRL/20200408/85293695R2.csv",index_col=1)

## 312個

## 先生方↓
data317=pd.read_csv("datasRL/Log/fujitaL1.csv",index_col=1)
data318=pd.read_csv("datasRL/Log/fujitaL2.csv",index_col=1)
data319=pd.read_csv("datasRL/Log/fujitaR1.csv",index_col=1)
data320=pd.read_csv("datasRL/Log/fujitaR2.csv",index_col=1)
data321=pd.read_csv("datasRL/Log/koyamaL1.csv",index_col=1)
data322=pd.read_csv("datasRL/Log/koyamaL2.csv",index_col=1)
data323=pd.read_csv("datasRL/Log/koyamaR1.csv",index_col=1)
data324=pd.read_csv("datasRL/Log/koyamaR2.csv",index_col=1)
data325=pd.read_csv("datasRL/Log/koyanagiL1.csv",index_col=1)
data326=pd.read_csv("datasRL/Log/koyanagiL2.csv",index_col=1)
data327=pd.read_csv("datasRL/Log/koyanagiR1.csv",index_col=1)
data328=pd.read_csv("datasRL/Log/koyanagiR2.csv",index_col=1)
data329=pd.read_csv("datasRL/Log/nimuraL1.csv",index_col=1)
data330=pd.read_csv("datasRL/Log/nimuraL2.csv",index_col=1)
data331=pd.read_csv("datasRL/Log/nimuraR1.csv",index_col=1)
data332=pd.read_csv("datasRL/Log/nimuraR2.csv",index_col=1)
data333=pd.read_csv("datasRL/Log/suzukiL1.csv",index_col=1)
data334=pd.read_csv("datasRL/Log/suzukiL2.csv",index_col=1)
data335=pd.read_csv("datasRL/Log/suzukiR1.csv",index_col=1)
data336=pd.read_csv("datasRL/Log/suzukiR2.csv",index_col=1)


def ffts(array, number_tree):
    data_contena = []
    target_contena = []
    y = array.iloc[:, number_tree]
    for i in range(len(y)-250, len(y), 10):
        y1 = y.iloc[i-128:i]
        y1 = np.array(y1)  # np.ndarray
        data_regista = np.zeros((nnn, a))
        yf = fft(y1)/(N/2)
        data_regista = yf[1:65]
        data_regista2 = data_regista.reshape(a*nnn)
        data_contena.append(data_regista2)
        if counter >= normal_counter:
            target_contena.append(0)
        else:
            target_contena.append(1)
    return data_contena, target_contena


def func1(lst, value):
    return [i for i, x in enumerate(lst) if x == value]


tree_list_box = []
score_list_box = []
for z in range(1):
    n = 1
    nnn = 1
    # parameters
    N = 128  # data number
    dt = 1/60  # data step [s]
    train_data_contena = []
    train_target_contena = []
    normal_counter = 21  # healthy people of number
    counter = 0
    human_list = [df0, df27, df2, df5, df3, df4, df22, df23, df24, df26, df27, df28, df31, df32, df33, df48, df49, df50, df51, df52, df53, df6_c, df7_c, df8_c, df9_c, df10_c,
                  df11_c, df12_c, df13_c, df14_c, df15_c, df35_c, df36_c, df37_c, df38_c, df39_c, df16_c, df17_c, df19_c, df18_c, df20_c, df21_c, df40_c,df45_c, df46_c, df47_c]
    a = 64
    mm = MinMaxScaler()

    first_score = 0
    first_vec = []
    second = []
    first_tree_num = 0
    for j in range(66):
        train_data_contena = []
        train_target_contena = []
        train_box = []
        target_box = []
        counter = 0
        a = 64
        for i in human_list:
            train_box = []
            target_box = []
            train_box, target_box = ffts(i, j)
            if counter == 0:
                reshape_base = np.array(target_box).size
            train_data_contena.append(np.array(train_box))
            train_target_contena.append(np.array(target_box))
            counter += 1

        counter = 0
        normal_counter = 4
        test_target_contena = []
        test_data_contena = []
        test_list = [df54, df55, df56, df57,df41_c, df42_c, df43_c, df44_c, ]
        test_data_regista = np.zeros((nnn, a))
        

        for i in test_list:
            train_box = []
            target_box = []
            train_box, target_box = ffts(i, j)
            if counter == 0:
                reshape_base = np.array(target_box).size
            test_data_contena.append(np.array(train_box))
            test_target_contena.append(np.array(target_box))
            counter += 1

        train_data_contena = np.array(np.abs(train_data_contena))
        test_data_contena = np.array(np.abs(test_data_contena))
        train_data_contena = train_data_contena.reshape(46*reshape_base, 1*a)
        test_data_contena = test_data_contena.reshape(8*reshape_base, 1*a)
        train_target_contena = np.array(
            train_target_contena).reshape(46*reshape_base)
        test_target_contena = np.array(
            test_target_contena).reshape(8*reshape_base)
        # train_data_contena2 = mm.fit_transform(train_data_contena)
        train_data_contena2 = train_data_contena
        clf = RandomForestClassifier()
        pca = PCA(n_components=2)
        X = pca.fit_transform(train_data_contena2)
        X_t = pca.fit_transform(test_data_contena)
        
        # stratifiedkfold = StratifiedKFold(n_splits=6)
        # pred_score = cross_val_score(
        #     clf, X, train_target_contena, cv=stratifiedkfold)
        # pred_score = pred_score.mean()
        # pred_vec = cross_val_predict(
        #     clf, X, train_target_contena, cv=stratifiedkfold)
        clf.fit(X,train_target_contena)
        # clf.predict(X_t)
        print("aaa")
        print( clf.predict(X_t))
        print( clf.score(X_t,test_target_contena))
        print("bbb")
       

    idx = func1(first_vec == train_target_contena, False)
    print(first_score, first_tree_num)
    second_target_contena = train_target_contena[idx]
    second_data_contena = X[idx]
    print(second_target_contena)
    print(second_data_contena)

    # step2

    second_tree_num = 0
    second_score = 0
    second_vec = []
    for j in range(66):
        pred_score = cross_val_score(
            clf, second_data_contena, second_target_contena, cv=stratifiedkfold)
        pred_score = pred_score.mean()
        pred_vec = cross_val_predict(
            clf, second_data_contena, second_target_contena, cv=stratifiedkfold)
        if second_score < pred_score:
            second_score = pred_score
            second_vec = pred_vec
            second_tree_num = j
    print(second_target_contena.shape)
    print(second_vec.shape)
    idx = func1(second_vec == second_target_contena, False)
    print(second_score)
    third_target_contena = second_target_contena[idx]
    third_data_contena = second_data_contena[idx]
    print(third_target_contena)
    print(third_data_contena)

    tree_list = [first_tree_num, second_tree_num]
    score_list = [first_score, second_score]
print(tree_list)
print(score_list)





