from matplotlib.patches import PathPatch, Patch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from xml.dom import minidom
from svgpath2mpl import parse_path
from dataclasses import dataclass


@dataclass
class SVGPath:
  path : Path
  style : dict

def parse_fill_atrib(value):
  if value == 'none':
    return {
      'fill': False,
    }
  return {
    'fill': True,
    'facecolor': value
  }

def parse_stroke_attrib(value):
  if value == 'none':
    return {
      'edgecolor': 'white'
    }
  return {
    'edgecolor': value
  }

def parse_stroke_width_attrib(value):
  return {
    'lw': float(value)
  }

def parse_fill_opacity_attrib(value):
  return {
    'alpha': float(value)
  }

def parse_style(style : str):
  attribs = {
    'fill': True,
    'facecolor': 'black'
  }
  for pair in style.split(';'):
    name,value = pair.split(':')
    match name:
      case 'fill':
        attribs = {**attribs, **parse_fill_atrib(value)}
      case 'stroke':
        attribs = {**attribs, **parse_stroke_attrib(value)}
      case 'stroke-width':
        attribs = {**attribs, **parse_stroke_width_attrib(value)}
      case 'fill-opacity':
        attribs = {**attribs, **parse_fill_opacity_attrib(value)}
      case 'stroke-opacity':
        pass
      case _:
        pass
  return attribs

def load_pathes(svgfile):
  doc = minidom.parse(svgfile)
  pathes = []
  for xmlpath in doc.getElementsByTagName('path'):
    pathes.append(SVGPath(
      style = parse_style(xmlpath.getAttribute('style')),
      path = parse_path(xmlpath.getAttribute('d'))
    ))
  return pathes

def load_path_collection(svgfile):
  pathes = load_pathes(svgfile)
  patches = [PathPatch(p.path, **p.style) for p in pathes]
  collection = PatchCollection(patches, match_original=True)
  return collection
