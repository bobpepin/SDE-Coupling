import xml.etree.ElementTree as ET
import numpy as np
import h5py
import os.path
import copy

XDMFSTR = """
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Grid" GridType="Uniform" id="grid">
      <Topology TopologyType="3DCoRectMesh" id="topology" Dimensions="5 100 1">
      </Topology>
      <Geometry Type="ORIGIN_DXDYDZ">
         <!-- Origin -->
         <DataItem Format="XML" Dimensions="3">
0.0 0.0 0.0
         </DataItem>
         <!-- DxDyDz -->
        <DataItem Format="XML" Dimensions="3">
1.0 1.0 1.0
        </DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>"""

XDMFATTR = """
      <Attribute AttributeType="Vector" Name="" Center="Node" id="attr">
	<DataItem ItemType="Uniform"
		  Format="HDF"
		  NumberType="Float"
		  Precision="4"
		  Dimensions="" id="item">
	  </DataItem>
      </Attribute>"""

class Xdmf3File(object):
    def __init__(self, fname, mode=None):
        (root, ext) = os.path.splitext(fname)
        self.h5path = root + '.h5'
        self.xdmfpath = root + (ext if ext else '.xmf')
        self.h5fp = h5py.File(self.h5path, mode)
        (self.xml, self.xmlids) = ET.XMLID(XDMFSTR)
        self.dim = (0, 0)

    def __enter__(self):
        self.h5fp.__enter__()
        return self
    
    def __exit__(self, *args):
        self.close()
        self.h5fp.__exit__(*args)

    def add_vector(self, name, data):
        n = data.shape[-1]
        dim = data.shape[0:-1]
        if len(dim) == 1:
            dim = (dim[0], 1)
        self.h5fp.create_dataset(name, data=data, dtype='f')
        (attrxml, ids) = ET.XMLID(XDMFATTR)
        ids['attr'].set('Name', name)
        ids['attr'].set('AttributeType', 'Vector')
        ids['item'].set('Dimensions', '%d %d %d' % (dim[0], dim[1], n))
        ids['item'].text = '%s:/%s' % (os.path.basename(self.h5fp.filename), name)
        self.xmlids['grid'].append(attrxml)
        self.dim = np.maximum(self.dim, dim)

    def add_scalar(self, name, data):
        dim = data.shape
        if len(dim) == 1:
            dim = (1, dim[0])
        self.h5fp.create_dataset(name, data=data, dtype='f')
        (attrxml, ids) = ET.XMLID(XDMFATTR)
        ids['attr'].set('Name', name)
        ids['attr'].set('AttributeType', 'Scalar')
        ids['item'].set('Dimensions', '1 %d %d' % dim)
        ids['item'].text = '%s:/%s' % (os.path.basename(self.h5fp.filename), name)
        self.xmlids['grid'].append(attrxml)
        self.dim = np.maximum(self.dim, dim)

        
    def add_attribute(self, name, data):
        self.h5fp.create_dataset(name, data=data, dtype='f')
        (attrxml, ids) = ET.XMLID(XDMFATTR)
        ids['attr'].set('Name', name)
        ids['item'].set('Dimensions', '1 %d %d' % data.shape)
        ids['item'].text = '%s:/%s' % (os.path.basename(self.h5fp.filename), name)
        self.xmlids['grid'].append(attrxml)
        self.dim = np.maximum(self.dim, data.shape)
        
    def close(self):
        self.h5fp.close()
        self.xmlids['topology'].set('Dimensions', '1 %d %d' % tuple(self.dim))
        et = ET.ElementTree(self.xml)
        et.write(self.xdmfpath)

