import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import os


if __name__=="__main__":
    #FLOAT SECTION
    sns.set_theme()
    os.chdir('D:\knasiotis\Documents\EDEMM\Thesis\Konstantinos-Nasiotis-Thesis-EDEMM\Data')
    df = pd.read_csv('boxesproblem3.csv', header=0)
    #df = df.melt(['Device','Epsilon'], var_name='Problem', value_name='Boxes')
    print(df)
    df = sqldf('''SELECT *, Boxes/Minimum AS Boxes2 ,LOG(Boxes/Minimum) AS Boxes3 FROM df ORDER BY Device,Problem,Epsilon DESC''')
    print(df)
    df = df.astype({'Epsilon':str})
    plotfloat = sns.lineplot(data=df, x="Epsilon", y='Boxes', hue='Problem', markers=True, style='Device', errorbar=None)
    #plotfloat.invert_xaxis()
    plotfloat.set_ylabel("Boxes")
    #plotfloat.set_xticks(epsilon1)
    #plotfloat.set_xticklabels(epsilon1.reverse())
    #plt.legend(title='Problem')
    plt.title('Boxes Generated')
    plt.show()

    plotfloat = sns.lineplot(data=df, x="Epsilon", y='Boxes3', hue='Problem', markers=True, style='Device', errorbar=None)
    #plotfloat.invert_xaxis()
    plotfloat.set_ylabel("Boxes")
    #plotfloat.set_xticks(epsilon1)
    #plotfloat.set_xticklabels(epsilon1.reverse())
    #plt.legend(title='Problem')
    plt.title('Rate of Input Space Expansion (Logarithmic Scale)')
    plt.show()