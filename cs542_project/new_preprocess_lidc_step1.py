import xmltodict
import numpy as np
from glob import glob

ANN_FILTER = './data_luna16/annotation_XMLonly/tcia-lidc-xml/*/*.xml'
SAVE_FILE = './data_luna16/lidc_xml_preprocessed.npy'

def load_xml_file(path):
    with open(path) as f:
        doc = xmltodict.parse(f.read())
    return doc

def dict2list(d):
    if isinstance(d, dict):
        d = [d]
    return d

def load_nodule_roi(roi):
    roi = dict2list(roi)

    layers = []
    for scan in roi:
        edge_map = dict2list(scan['edgeMap'])

        pts = []
        for pt in edge_map:
            x_pix, y_pix = pt['xCoord'], pt['yCoord']
            pts.append([x_pix, y_pix])

        layers.append({'imageZposition': scan['imageZposition'],
                       'edgeCoords': pts})
    return layers

def read_instance(xml_path):
    xml_obj = load_xml_file(xml_path)

    try:
        message = xml_obj['LidcReadMessage']
    except:
        message = xml_obj['IdriReadMessage']

    # The uid of the patient
    try:
        uid = message['ResponseHeader']['SeriesInstanceUid']
    except:
        uid = message['ResponseHeader']['SeriesInstanceUID']

    # Read each reading (there's usually four)
    try:
        reading_sessions = message['readingSession']
    except:
        if 'CXRreadingSession' in message:
            reading_sessions = message['CXRreadingSession']
        else:
            print('no reading session in', xml_path)
            return {'uid': uid, 'readings': []}
    # only a few examples have empty reading session:
    if not reading_sessions:
        print('empty reading session in', xml_path)
        return {'uid': uid, 'readings': []}

    assert(isinstance(reading_sessions, list))
    print('number of reading sessions:', len(reading_sessions))

    readings = []
    # read each nodule in each reading session
    for r in reading_sessions:
        if 'unblindedReadNodule' not in r:
            continue
        nodule_info_list = r['unblindedReadNodule']
        if isinstance(nodule_info_list, dict):
            nodule_info_list = [nodule_info_list]
        for nodule_info in nodule_info_list:
            # those nodules < 3 does not have 'characteristics' key
            if 'characteristics' in nodule_info and nodule_info['characteristics']:
                roi = load_nodule_roi(nodule_info['roi'])
                malignancy = int(nodule_info['characteristics']['malignancy'])
                readings.append({'roi': roi, 'malignancy': malignancy})

    ann = {'uid': uid, 'readings': readings}
    return ann

xml_files = glob(ANN_FILTER)

xml_anns = [None]*len(xml_files)
for n, xml_path in enumerate(xml_files):
    xml_anns[n] = read_instance(xml_path)

np.save(SAVE_FILE, np.array(xml_anns))
