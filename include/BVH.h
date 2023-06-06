#ifndef _BVH_H_
#define _BVH_H_
#include <iostream>
#include <vector>
#include "Triangle.h"
#include "Eigen/Dense"
#include "Global.h"

struct BVHNode
{
    int lc, rc;
    unsigned int n;
    int it;
    Eigen::Vector3f AA, BB;
    BVHNode():lc(-1), rc(-1), it(-1), n(0)
    {
        AA = Eigen::Vector3f(MAXF, MAXF, MAXF);
        BB = Eigen::Vector3f(MINF, MINF, MINF);
    }
};

class BVH
{
    private:
        int root_index;
        unsigned int thresh_n;
        std::vector<Triangle>& triangles;
        std::vector<BVHNode> nodes;
    public:
        BVH(const unsigned int n, std::vector<Triangle> &ts):thresh_n(n), root_index(-1), triangles(ts)
        {
            nodes.clear();
            root_index = build_node(0, (int)triangles.size());
            // printf("Triangles size = %llu\n", triangles.size());
            // for(int i=0;i<nodes.size();i++)
            // {
            //     printf("Triangle index begin = %d, end = %d\n", nodes[i].it, nodes[i].it+nodes[i].n);
            // }
        };

    private:
        int build_node(int l, int r)
        {
            if(l >= r)
                return -1;
            BVHNode node;
            // set AA, BB
            for(int i = l; i < r; i++)
            {
                node.AA.x() = std::min(triangles[i].get_min_x(), node.AA.x());
                node.AA.y() = std::min(triangles[i].get_min_y(), node.AA.y());
                node.AA.z() = std::min(triangles[i].get_min_z(), node.AA.z());

                node.BB.x() = std::max(triangles[i].get_max_x(), node.BB.x());
                node.BB.y() = std::max(triangles[i].get_max_y(), node.BB.y());
                node.BB.z() = std::max(triangles[i].get_max_z(), node.BB.z());
            }

            // terminate or not
            node.it = l;
            node.n = r - l;
            if(node.n <= thresh_n)
            {
                nodes.push_back(node);
                return static_cast<int>(nodes.size())-1;
            }

            // partition
            Eigen::Vector3f delta = node.BB - node.AA;
            if(delta.x() >= delta.y() && delta.x() >= delta.z())
            {
                std::sort(triangles.begin()+l, triangles.begin()+r, Triangle::cmp_x);
            }
            else if(delta.y() >= delta.x() && delta.y() >= delta.z())
            {
                std::sort(triangles.begin()+l, triangles.begin()+r, Triangle::cmp_y);
            }
            else if(delta.z() >= delta.x() && delta.z() >= delta.y())
            {
                std::sort(triangles.begin()+l, triangles.begin()+r, Triangle::cmp_z);
            }

            // set left & right child
            int mid = (l + r) / 2;
            node.lc = build_node(l, mid);
            node.rc = build_node(mid, r);
            nodes.push_back(node);
            return (int)nodes.size()-1;
        }
    public:
        int get_root_index() const
        {
            return root_index;
        }

        size_t get_nodes_size() const
        {
            return nodes.size();
        }

        const std::vector<BVHNode>& get_nodes()
        {
            return nodes;
        }

        const std::vector<Triangle>& get_triangles()
        {
            return triangles;
        }

        BVHNode* get_nodes_data()
        {
            return nodes.data();
        }

        size_t get_triangles_size() const
        {
            return triangles.size();
        }

        Triangle* get_triangles_data()
        {
            return triangles.data();
        }
};
#endif