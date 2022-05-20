import networkx as nx
import matplotlib.pyplot as plt


def draw_graphs():
  T = nx.gnp_random_graph(16, 0.25)
  Q = nx.gnp_random_graph(8, 0.25)
  plt.figure(1)
  nx.draw(T)
  plt.figure(2)
  nx.draw(Q)
  print ("Before plt.show")
  plt.show()
  print ("After plt.show")

def other_stuff():
  print ("Other stuff!")


draw_graphs()
other_stuff()    
