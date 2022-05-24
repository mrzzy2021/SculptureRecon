#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import trimesh
import numpy as np
import argparse
import os


def computer_metrics(pred,target):

    pred = trimesh.load(pred)
    target = trimesh.load(target)
    chamfer_loss = computer_chamfer_distance(pred,target)
    p2s = computer_surface_dist(pred,target)

    return chamfer_loss,p2s

def computer_surface_dist(src,tgt, num_samples=10000):
    #P2S distance 
    src_surf_pts, _ = trimesh.sample.sample_surface(src, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()

    return src_tgt_dist

def computer_chamfer_distance(src,tgt,num_samples = 10000):

    src_surf_pts, _ = trimesh.sample.sample_surface(src, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt, num_samples)
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src, tgt_surf_pts)
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0
    src_tgt_dist = (src_tgt_dist).mean()
    tgt_src_dist = (tgt_src_dist).mean()
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    return chamfer_dist


if __name__ == '__main__':
    
    chamfer = 0
    p = 0
    
    for i in range(4,5):
            obj = '%02d' % i
            ground_truth = '../' + str(obj) + '.obj'
            pred = '../' + str(obj) + '.obj'
            chamfer_loss,p2s = computer_metrics(pred,ground_truth)
            chamfer = chamfer + chamfer_loss
            p = p + p2s







