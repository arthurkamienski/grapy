import numpy as np
import pandas as pd

class Node:
    def __init__(self, atts):
        self.__dict__ = atts.copy()
        self.atts = atts
        
        self.edges_in = {}
        self.edges_out = {}
        
    def __getitem__(self, item):
        if type(item) == list:
            return {i: self.atts[i] for i in item}
        else:
            return self.atts[item]
        
    def add_out_edge(self, other, atts):
        self.edges_out[other] = atts
        
    def add_in_edge(self, other, atts):
        self.edges_in[other] = atts

    def __repr__(self):
        return 'Node' + str(self.atts)

class Graph:
    def __init__(self, nodes=None):
        self.nodes = {}
        self.edge_num = 0
        self.atts_nodes = {}
        self.atts_edges = {}

        if nodes is not None:
            self.nodes = nodes
            self.remove_dangling()

    def to_list(self, edges=False, columns=None):
        nodes = list(self.nodes.values())

        if edges:
            if columns is None:
                i = 0
                while len(nodes[i].edges_in) == 0:
                    i += 1
                columns = list(nodes[i].edges_in.values())[0].keys()
            rows = [[e[c] for c in columns] for n in nodes for e in n.edges_in.values()]
        else:
            if columns is None:
                columns = nodes[0].atts.keys()
            rows = [[n.atts[c] for c in columns] for n in nodes]

        return list(columns), rows

    def to_dataframe(self, edges=False, columns=None):
        header, rows = self.to_list(edges, columns)
        return pd.DataFrame(rows, columns=header)


    def to_gdf(self, filename, atts_nodes=None, atts_edges=None, edges_in=True):
        atts_nodes, nodes = self.to_list(columns=atts_nodes)
        atts_edges, edges = self.to_list(edges=True, columns=atts_edges)
        
        atts_nodes = [f'{a} {gdf_type[type(v)]}' for a, v in zip(atts_nodes, nodes[0])]
        atts_edges = [f'{a} {gdf_type[type(v)]}' for a, v in zip(atts_edges, edges[0])]

        with open(filename, 'w') as f:
            f.write('nodedef>' + ','.join(atts_nodes) + '\n')
            for n in nodes:
                f.write(','.join(map(str, n)) + '\n')
            
            f.write('edgedef>' + ','.join(atts_edges) + '\n')
            for e in edges:
                f.write(','.join(map(lambda x: str(x).lower() if type(x) == bool else str(x), e)) + '\n')

    def apply(self, func):
        for n in self.nodes.values():
            func(n)

    def filter_nodes(self, func):
        self.nodes = {n.name: n for n in list(self.nodes.values()) if func(n)}
        self.remove_dangling()

    def remove_dangling(self):
        for n in self.nodes.values():
            n.edges_out = [e for e in n.edges_out if e.name in self.nodes]
            n.edges_in = [e for e in n.edges_in if e.name in self.nodes]

    def add_nodes(self, nodes):
        self.nodes = {n['name']: Node(n) for n in nodes}

    def add_edges(self, edges):
        self.edge_num += len(edges)

        for e in edges:
            node1 = self.nodes[e['node1']]
            node2 = self.nodes[e['node2']]
            
            node1.add_out_edge(node2, e)
            node2.add_in_edge(node1, e)

    def node(self, node):
        return self.nodes[node]
            
    def __getitem__(self, item):
        if callable(item):
            return {n.name: n for n in list(self.nodes.values()) if item(n)}
        else:
            return {n.name: n[item] for n in list(self.nodes.values())}
            
    def __repr__(self):
        return f'Graph({len(self.nodes)}, {self.edge_num})'


class gdf():
    to_dtype = {
        'VARCHAR': str,
        'INTEGER': int,
        'BOOLEAN': (lambda x: x=='true'),
        'DOUBLE': float,
    }

    type_of = {
        str: 'VARCHAR',
        int: 'INTEGER',
        bool: 'BOOLEAN',
        float: 'DOUBLE',
        np.dtype('O'): 'VARCHAR',
        np.dtype('int64'): 'INTEGER',
        np.dtype('bool'): 'BOOLEAN',
        np.dtype('float64'):'DOUBLE'
    }

    def read(filepath):
        def read_file(filepath):
            with open(filepath, 'r') as file:
                content = file.read()

                pos_edgedef = content.find('edgedef>')

                nodedef = content[:pos_edgedef]
                edgedef = content[pos_edgedef:]

            def split(csv):
                csv = csv[8:].strip().split('\n', 1)
                header = [h.strip().split(' ') for h in csv[0].split(',')]
                header = {name: tp for name, tp in header}
                lines = [l.split(',') for l in csv[1].split('\n')]
                
                return header, lines

            return split(nodedef), split(edgedef)
        
        def to_list(header, lines):
            types = [(name, gdf.to_dtype[tp]) for name, tp in header.items()]
            
            def to_dict(line):
                d = {}

                for t, v in zip(types, line):
                    name, tp = t
                    try:
                        d[name] = tp(v)
                    except ValueError:
                        d[name] = np.nan
                return d
            
            return [to_dict(l) for l in lines]

        nodedef, edgedef = read_file(filepath)

        return to_list(*nodedef), to_list(*edgedef)

    def from_df(filename, ns, es, gdf_header=True):
        def save_to_file(df):
            def gdf_header(df):
                prefix = 'edgedef>' if 'node1' in df.columns else 'nodedef>'
                types = [gdf.type_of[dt] for dt in df.dtypes]
                col_names = [f"{n} {t}" for n, t in zip(df.columns, types)]
                col_names[0] = prefix + col_names[0]
                return ','.join(col_names)
            
            mode = 'a' if 'node1' in df.columns else 'w'

            if gdf_header:
                header = gdf_header(df)

                with open(filename, mode) as file:
                    file.write(header + '\n')
                    
                mode='a'

            df.to_csv(filename, header=False, index=False, mode=mode)

        save_to_file(ns)
        save_to_file(es)

    def to_df(filepath):
        ns, es = gdf.read(filepath)

        return pd.DataFrame(ns), pd.DataFrame(es)

def from_gdf(filepath):
    nodes, edges = read_gdf(filepath)

    g = Graph()
    g.atts_nodes = nodes[0]
    g.atts_edges = edges[0]
    
    g.add_nodes(nodes[1])
    g.add_edges(edges[1])

    return g