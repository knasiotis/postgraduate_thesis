import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import os
epsilon1 = ['1e-2','1e-3','1e-4','1e-5']
epsilon2 = ['0.1','0.06','0.03']
if __name__=="__main__":
    #FLOAT SECTION
    sns.set_theme()
    os.chdir('D:\knasiotis\Documents\EDEMM\Thesis\Konstantinos-Nasiotis-Thesis-EDEMM\Data')
    df = pd.read_csv('raw_data_griewank.csv', header=0)
    cpu = sqldf('''SELECT * FROM df WHERE DEVICE="CPU"''')
    #gnet float speedup
    dfloat = sqldf('''SELECT * FROM 
                   (SELECT df.*, cpu.Time/df.Time AS Speedup FROM df 
                   INNER JOIN cpu ON df.Epsilon==cpu.Epsilon AND df.Type==cpu.Type 
                   GROUP BY df.Epsilon,df.Device, df.Type 
                   ORDER BY df.Device,df.Epsilon) 
                   WHERE Device!='CPU' AND Type='float' ''')
    dfloat = dfloat.astype({'Epsilon':str})
    print(dfloat)
    plotfloat = sns.lineplot(data=dfloat, x="Epsilon", y="Speedup", hue='Device', markers=True, style='Device', errorbar=None)
    plotfloat.invert_xaxis()
    #plotfloat.set_xticks(epsilon1)
    #plotfloat.set_xticklabels(epsilon1.reverse())
    plt.legend(title='GPU Configuration')
    plt.title('Problem 2: Speedup vs 5820K(1 Core) FP32')
    plt.show()
    
    dfloat_throughput = sqldf(''' SELECT *, LOG(Boxes/Time) AS Throughput FROM df WHERE Type='float' ''')
    dfloat_throughput = dfloat_throughput.astype({'Epsilon':str})
    print(dfloat_throughput)
    plotfloat = sns.lineplot(data=dfloat_throughput, x="Epsilon", y="Throughput", hue='Device', markers=True, style='Device', errorbar=None)
    #plotfloat.invert_xaxis()
    plt.ylim(0,10)
    plt.legend(title='GPU Configuration')
    #sns.move_legend(plotfloat, "best", bbox_to_anchor=(1, 1))
    plt.title('Problem 2: Throughput FP32 (Logarithmic Scale)')
    plt.show()


    df_float_runs = sqldf('''SELECT * FROM df where Device!='CPU' AND Type='float' ''')
    dfloat_runs = sns.catplot(data=df_float_runs, kind="bar",x="Runs", y="Device", hue="Epsilon",errorbar=None)
    #plt.title('Problem 1: Number of Runs FP32', y=-0.01)
    #plt.xticks(rotation="vertical")
    plt.show()

    #HALF SECTION
    """
    #gnet half speedup
    dhalf = sqldf('''SELECT * FROM 
                  (SELECT df.*, cpu.Time/df.Time AS Speedup FROM df 
                  INNER JOIN cpu ON df.Epsilon==cpu.Epsilon AND df.Type==cpu.Type 
                  GROUP BY df.Epsilon,df.Device,
                  df.Type ORDER BY df.Device,df.Epsilon) 
                  WHERE Device!='CPU' AND Type='half' ''')
    plothalf = sns.lineplot(data=dhalf, x="Epsilon", y="Speedup", hue='Device', markers=True, style='Device', errorbar=None).invert_xaxis()
    plt.legend(title='GPU Configuration')
    plt.title('Problem 3: Speedup vs 5820K(1 Core) FP16')
    plt.show()

    dhalf_throughput = sqldf(''' SELECT *, LOG(Boxes/Time) AS Throughput FROM df WHERE Type='half' ''')
    print(dhalf_throughput)
    plothalf = sns.lineplot(data=dhalf_throughput, x="Epsilon", y="Throughput", hue='Device', markers=True, style='Device', errorbar=None)
    plothalf.invert_xaxis()
    plt.ylim(0,10)
    plt.legend(title='GPU Configuration')
    #sns.move_legend(plotfloat, "best", bbox_to_anchor=(1, 1))
    plt.title('Problem 3: Throughput FP16 (Logarithmic Scale)')
    plt.show()


    df_half_runs = sqldf('''SELECT * FROM df where Device!='CPU' ''')
    dhalf_runs = sns.catplot(data=df_half_runs, kind="bar",x="Device", y="Runs", hue="Epsilon",errorbar=None)
    plt.title('Problem 3: Number of Runs FP32 Logarithmic Scale')
    plt.show()
    """