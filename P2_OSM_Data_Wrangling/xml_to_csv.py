#!/usr/bin/env python
# -*- coding: utf-8 -*-



# code base from Lesson 13 - Preparing for Database
# adapted for Python 3

import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema1

OSM_PATH = "map_hamburg_stade.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema1.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


BETTER_KEYS = {
        'postal_code': 'postcode',
        'postal_codes': 'postcode',
        'streetnumber': 'housenumber'
        }

def split_address(tag_dict):
    '''
    splits one tag with key 'addr' in its single parts and creates a dict
    for each of them
    '''
    tags = []
    tag_dict['type'] = tag_dict['type'] + ':addr' # set tag type 
    addr_list = tag_dict['value'].split(',')
    # first: part one, street and housenumber
    if re.search(r'\d+', addr_list[0]) != None and ' ' in addr_list[0]: # checks if housenumber included and handles bad Unicode exception 
        # last part is housenumber, rest street
        housenumber = addr_list[0].rsplit(' ', 1)[1]
        housenumber_dict = tag_dict
        housenumber_dict['key'] = 'housenumber'
        housenumber_dict['value'] = housenumber
        if housenumber_dict not in tags: #check for duplicates
            tags.append(housenumber_dict)
                        
        street = addr_list[0].rsplit(' ', 1)[0]
        
    else:
        street = addr_list[0]
        if '/' in street:
            street = street.split('/')[0] # handles bad Unicode exception
        
    street_dict = tag_dict
    street_dict['key'] = 'street'
    street_dict['value'] = street
    if street_dict not in tags:
        tags.append(street_dict)
               
    if len(addr_list) == 2: # True when there has been a comma 
        if re.search(r'\d+', addr_list[1]) != None: # checks if postcode included
            # first part is postcode, second the city
            postcode = addr_list[1].split(' ', 1)[0]
            postcode_dict = tag_dict
            postcode_dict['key'] = 'postcode'
            postcode_dict['value'] = postcode
            if postcode_dict not in tags:
                tags.append(postcode_dict)
                        
            city = addr_list[1].rsplit(' ', 1)[1]
            
        else:
            city = addr_list[1]
        
        city_dict = tag_dict
        city_dict['key'] = 'city'
        street_dict['value'] = city
        if city_dict not in tags:
            tags.append(city_dict)
    return tags
    
    
    
    
    
def handle_tags(element, id):
    '''
    creates a tags list with dicts for each tag of a way or node
    '''
    tags = [] 
    for child in element:
        tag_dict = {}
        if child.tag == 'tag':
            key = child.attrib['k']
            if PROBLEMCHARS.search(key):
                continue
            tag_dict['id'] = id
            if ':' in key:
                tag_dict['key'] = key.rsplit(":", 1)[1]
                tag_dict['type'] = key.rsplit(":", 1)[0]
                
            else:
                tag_dict['key'] = key
                tag_dict['type'] = 'regular'
            if tag_dict['key'] in BETTER_KEYS:
                tag_dict['key'] = BETTER_KEYS[tag_dict['key']]
            tag_dict['value'] = child.attrib['v']
            
            if tag_dict['key'] == 'addr':
                if element.tag == 'way':
                    pass    # drop tag  
                    
                else:
                   tags.extend(split_address(tag_dict))
                
            else:
                if tag_dict not in tags:
                    tags.append(tag_dict)
    return tags
    

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []

    if element.tag == 'node':
        node_full_attribs = element.attrib
        for key, value in node_full_attribs.items():
            if key in NODE_FIELDS:
                node_attribs[key] = value
        tags = handle_tags(element, node_attribs['id'])
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        way_full_attribs = element.attrib
        for key, value in way_full_attribs.items():
            if key in WAY_FIELDS:
                way_attribs[key] = value
        tags = handle_tags(element, way_attribs['id'])
        position = 0
        for child in element:
            if child.tag ==   'nd':
                nodes_dict = {
                    'id': way_attribs['id'],
                    'node_id': child.attrib['ref'],
                    'position': position
                }
                position += 1
                way_nodes.append(nodes_dict)
                
            
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

'''
class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.items()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
'''

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w', encoding='utf-8-sig') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w', encoding='utf-8-sig') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w', encoding='utf-8-sig') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w', encoding='utf-8-sig') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w', encoding='utf-8-sig') as way_tags_file:

        nodes_writer = csv.DictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = csv.DictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = csv.DictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = csv.DictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = csv.DictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSM_PATH, validate=False)
