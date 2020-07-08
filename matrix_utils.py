import scipy.sparse as sp
import pandas as pd
import numpy as np
import networkx as nx
from operator import itemgetter
from python_utils import file_utils as fx



@fx.timer
def get_sparse_mx(df, fields, count):

    row_ = list(sorted(df[fields[0]].unique()))
    col_ = list(sorted(df[fields[1]].unique()))

    row = df[fields[0]].astype(pd.api.types.CategoricalDtype(categories=row_)).cat.codes
    col = df[fields[1]].astype(pd.api.types.CategoricalDtype(categories=col_)).cat.codes

    data = df[count].tolist()

    sparse_matrix = sp.csr_matrix((data, (row, col)), shape=(len(row_), len(col_)))

    df = pd.SparseDataFrame([pd.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0)
                            for i in np.arange(sparse_matrix.shape[0])],
                            index=row_, columns=col_, default_fill_value=0)
    return df


# @fx.timer
def get_square_adjacency_mx(df):

    cols = df.columns
    X = sp.csr_matrix(df.astype(int, errors='ignore').values)
    Xc = X.T * X  # multiply sparse matrix
    Xc.setdiag(0)  # reset diagonal
    # df = sp.coo_matrix(Xc)

    # create dataframe from co-occurence matrix in dense format
    df = pd.DataFrame(Xc.todense(), index=cols, columns=cols)

    return df


def get_edges(df):

    df = df.astype(int, errors='ignore')
    df = df.stack().reset_index()
    df.columns = ['source', 'target', 'weight']
    df = df[df['weight'] != 0]

    return df


# read csv files from folder and convert them to gephi friendly spreadsheets
def csv_to_gephi(folder):
    files = fx.get_fnames(folder)

    for f in files:
        fname = f + '.csv'
        df = pd.read_csv(folder + '/' + fname, sep=',')
        df = df.set_index(df.columns[0])
        mx = get_square_adjacency_mx(df)
        gx = get_edges(mx)
        gx.to_csv(folder + '/gephi_' + fname, sep=',', encoding='utf-8', index=False)


def get_graph(df):
    g = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])
    return g


def draw_graph(df, title):

    plt.figure(figsize=(20, 20))

    # create the graph
    g = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])

    # create the layout for the nodes
    layout = nx.spring_layout(g, iterations=10)

    # draw the parts we want
    nx.draw_networkx_edges(g, layout, edge_color='#AAAAAA')

    targets = [node for node in g.nodes() if node in df.target.unique()]
    size = [g.degree(node) * 5 for node in g.nodes() if node in df.target.unique()]

    nx.draw_networkx_nodes(g, layout, nodelist=targets, node_size=size, node_color='lightblue')

    sources = [node for node in g.nodes() if node in df.source.unique()]
    nx.draw_networkx_nodes(g, layout, nodelist=sources, node_size=50, node_color='#AAAAAA')

    high_degree_sources = [node for node in g.nodes() if node in df.source.unique() and g.degree(node) > 1]
    nx.draw_networkx_nodes(g, layout, nodelist=high_degree_sources, node_size=50, node_color='#fc8d62')

    target_dict = dict(zip(targets, targets))
    nx.draw_networkx_labels(g, layout, labels=target_dict)

    plt.axis('off')

    plt.title(title)

    plt.show()


def create_graph(nodes, edges, filename, scale):

    plt.figure(figsize=(20,20))

    G = nx.Graph()
    for node in nodes:
        G.add_node(node)

    edge_list = zip(edges['Source'], edges['Target'])

    G.add_edges_from(edge_list)

    # for (a, b), val in zip(edge_list, edges['Label'].values):
    #     G[a][b]['label'] = val


    node_color = [float(G.degree(v)) for v in G]


    pos = nx.random_layout(G)

    # use one of the edge properties to control line thickness
    edgewidth = edges['Weight']

    # discover triangles
#    print sorted(nx.triangles(G), reverse=True)
#    print sorted(nx.triangles(G).values(), reverse=True)

    nx.draw_networkx_nodes(G, pos, node_size=[float(G.degree(v)) * 200 for v in G], alpha=0.85, node_color=node_color, linewidths=0)
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color='#0EA6EC', width=[w * scale for w in edgewidth], arrows=False)

#    arquivo edge_labels = {'{}'.format(i[2]['label']) for i in G.edges(data=True)}
#    edge_labels = nx.get_edge_attributes(G,'label')
#
#    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

    # node_labels = {i:'{}'.format(i) for i in G.nodes()}
    # nx.draw_networkx_labels(G, pos, labels = node_labels, font_color='white', font_weight='bold')

    axes = plt.gca()
    axes.set_axis_bgcolor('#f5f5f5')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    plt.savefig(filename, dpi=150)
    plt.show

    return G


def create_community_graph(nodes, edges, filename, scale):

    import community.community_louvain as community
    plt.figure(figsize=(20,20))

    G = nx.Graph()

    for node in nodes:
        G.add_node(node)

    edge_list = zip(edges['Source'], edges['Target'])

    G.add_edges_from(edge_list)

    for (a, b), val in zip(edge_list, edges['Label'].values):
        G[a][b]['label'] = val


    pos = nx.circular_layout(G, scale=50 )

    # use one of the edge properties to control line thickness
    edgewidth = edges['Weight']

    parts = community.best_partition(G)
    node_color = [parts.get(node) for node in G.nodes()]

    print("Louvain Modularity: ", community.modularity(parts, G))

#    nx.draw_networkx(G, pos = pos, cmap = plt.get_cmap("jet"), node_color = node_color, node_size = [float(G.degree(v)) * 200 for v in G])
    nx.draw_networkx_nodes(G, pos, cmap = plt.get_cmap("rainbow"), node_size=[float(G.degree(v)) * 700 for v in G], alpha=0.7, node_color=node_color, linewidths=0)
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color='#0EA6EC', width=[w * scale for w in edgewidth], arrows=False)

#    arquivo edge_labels = {'{}'.format(i[2]['label']) for i in G.edges(data=True)}
#    edge_labels = nx.get_edge_attributes(G,'label')

#    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

    node_labels = {i:'{}'.format(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels = node_labels, font_color='#f5f5f5', font_weight='bold')

    axes = plt.gca()
    axes.set_axis_bgcolor('#f5f5f5')
#    plt.axis('off')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    plt.savefig(filename, dpi=150)
    plt.show

    return G


def get_degree_centrality(g):
    data = []
    dc = nx.degree_centrality(g)
    nx.set_node_attributes(g, values=dc, name='degree_cent')
    degcent_sorted = sorted(dc.items(), key=itemgetter(1), reverse=True)
    for key, value in degcent_sorted:
        data.append((key, value))
    df = pd.DataFrame(data)
    df.columns = ['node', 'degree_centrality']
    return df


def get_betweenness_centrality(g):
    data = []
    bc = nx.betweenness_centrality(g)
    betcent_sorted = sorted(bc.items(), key=itemgetter(1), reverse=True)
    for key, value in betcent_sorted:
        data.append((key, value))
    df = pd.DataFrame(data)
    return df


def topTable(field1, field2, n_top):
    topM = max(field2) * 0.9
    right = len(field1) * 0.75
    plt.text(right, topM * 1.08, 'Top %s' % n_top, fontsize=12)
    for i in range(n_top):
        curr = field1[i]
        val = field2[i]
        plt.text(right, topM - i * topM / 20, '{}) {} = {}'.format(i + 1,
        curr.upper(), round(val, 3)), fontsize=10)


def get_average_degree(G):
    N = G.order()
    K = G.size()
    avg_d = float(N) / K
    return avg_d


def get_network_density(G):
    return nx.density(G)


def plot_metrics(degc_key, degc_value, betc_key, betc_value, avg_degree, filename):
    # Plot: Degree_centrality
    plt.figure(figsize=(20,20))

    ax1 = plt.subplot(211)
#    plt.title('Degree centrality for nodes', fontsize=12)
    a_lenght = np.arange(len(degc_value))
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.labelpad = 50
    ax1.yaxis.labelpad = 50
    plt.bar(a_lenght, degc_value, color=cm.jet(degc_value), align='center', edgecolor = "none")
    plt.xticks(a_lenght, degc_key, size='small', rotation='vertical')
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='both', which='both',length=0)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.savefig(filename, dpi=150)
    plt.show()


    #Top degree centrality:
#    topTable(degc_key, degc_value, 10)
#    plt.text(len(degc_value) * 0.75, max(degc_value) * 0.4, avg_degree,
#    bbox={'facecolor': '#0EA6EC', 'alpha': 1, 'pad': 15}, fontsize=10)

    # Plot: Betweenness_centrality
#    plt.subplot(212)
#    plt.title('Betweenness centrality for nodes', fontsize=12)
#    a_lenght = np.arange(len(betc_value))
#    plt.bar(a_lenght, betc_value, color=cm.jet(betc_value), align='center')
#    plt.xticks(a_lenght, betc_key, size='small', rotation='vertical')
#    plt.tick_params(axis='x', labelsize=10)
#    plt.tick_params(axis='y', labelsize=10)
#    plt.autoscale(enable=True, axis='both', tight=None)
#    plt.ylim(0, max(betc_value) * 1.1)
#    plt.plot(betc_value, '--b')


def print_Top(nodes, df, top):

    n_array=[]

    # find top nodes
    for i in range(0, top):
        n_array.append(nodes[i])

    # remove the other nodes from the dataframe
    for j in df.index.values:
        if j not in n_array:
            df = df.drop(df[df.index == j].index )

    for row_index, row in df.iterrows():
       # iterate through all elements in the row
       print('\n' + str(row.name ))
       for colname in df.columns:
           row_element = row[colname]
           if row_element > 0:
               print(colname, row_element)



#T = create_graph(trends_nodes, trends_edges, 'trends_graph.eps', 0.5)

#G = create_graph(guardian_nodes, guardian_edges, 'guardian_graph.png', 3)

# TT = create_graph(trends_nodes_tr, trends_edges_tr, 'trends_graph.png', 0.5)
# #
# GT = create_graph(guardian_nodes_tr, guardian_edges_tr, 'guardian_graph.eps', 1)
#
# # generate metrics
#
# trends_degc_key, trends_degc_value = calculate_degree_centrality(TT)
# trends_betc_key, trends_betc_value = calculate_betweenness_centrality(TT)
# #trends_ad = average_degree(TT)
# ###

#plot_metrics(trends_degc_key, trends_degc_value, trends_betc_key, trends_betc_value, trends_ad, 'trends_metrics.pdf')
#plot_metrics(guardian_degc_key, guardian_degc_value, guardian_betc_key, guardian_betc_value, guardian_ad, 'guardian_metrics.pdf')


# print top nodes

#print_Top(trends_degc_key, trends_df, 10)
#print_Top(guardian_degc_key, guardian_df, 10)

# find correlations

#print np.corrcoef(trends_betc_value, guardian_betc_value)
#print np.corrcoef(trends_degc_value, guardian_degc_value)

# community detection

#CG = create_community_graph(guardian_nodes_tr, guardian_edges_tr, 'guardian_community.pdf', 1)
#TG = create_community_graph(trends_nodes_tr, trends_edges_tr, 'trends_community.pdf', 0.5)


#guardian_df.to_csv('matrix_2.csv', sep=',', encoding='utf-8')
# gf = pd.DataFrame()
# gf['terms'] = guardian_degc_key
# gf['values'] = guardian_degc_value
#
#
# tf = pd.DataFrame()
# tf['terms'] = trends_degc_key
# tf['trends'] = trends_degc_value
#
# zf = pd.merge(tf, gf, on='terms')
# #
# print np.round(zf.corr(), 2)
