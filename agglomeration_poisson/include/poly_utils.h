// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------


#ifndef poly_utils_h
#define poly_utils_h


#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/std_cxx20/iota_view.h>

#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/boost_adaptors/point.h>
#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/index/detail/rtree/utilities/print.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>

#include <deal.II/cgal/point_conversion.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <EpetraExt_RowMatrixOut.h>
#endif

#ifdef DEAL_II_WITH_CGAL

#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Constrained_triangulation_plus_2.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Segment_Delaunay_graph_2.h>
#  include <CGAL/Segment_Delaunay_graph_traits_2.h>
#  include <CGAL/intersections.h>
#  include <CGAL/squared_distance_2.h>
#  include <CGAL/squared_distance_3.h>


#endif

#include <memory>


namespace dealii::PolyUtils::internal
{
  /**
   * Helper function to compute the position of index @p index in vector @p v.
   */
  inline types::global_cell_index
  get_index(const std::vector<types::global_cell_index> &v,
            const types::global_cell_index               index)
  {
    return std::distance(v.begin(), std::find(v.begin(), v.end(), index));
  }



  /**
   * Compute the connectivity graph for locally owned regions of a distributed
   * triangulation.
   */
  template <int dim, int spacedim>
  void
  get_face_connectivity_of_cells(
    const parallel::fullydistributed::Triangulation<dim, spacedim>
                                               &triangulation,
    DynamicSparsityPattern                     &cell_connectivity,
    const std::vector<types::global_cell_index> locally_owned_cells)
  {
    cell_connectivity.reinit(triangulation.n_locally_owned_active_cells(),
                             triangulation.n_locally_owned_active_cells());


    // loop over all cells and their neighbors to build the sparsity
    // pattern. note that it's a bit hard to enter all the connections when
    // a neighbor has children since we would need to find out which of its
    // children is adjacent to the current cell. this problem can be omitted
    // if we only do something if the neighbor has no children -- in that
    // case it is either on the same or a coarser level than we are. in
    // return, we have to add entries in both directions for both cells
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            const unsigned int index = cell->active_cell_index();
            cell_connectivity.add(get_index(locally_owned_cells, index),
                                  get_index(locally_owned_cells, index));
            for (auto f : cell->face_indices())
              if ((cell->at_boundary(f) == false) &&
                  (cell->neighbor(f)->has_children() == false) &&
                  cell->neighbor(f)->is_locally_owned())
                {
                  const unsigned int other_index =
                    cell->neighbor(f)->active_cell_index();

                  cell_connectivity.add(get_index(locally_owned_cells, index),
                                        get_index(locally_owned_cells,
                                                  other_index));
                  cell_connectivity.add(get_index(locally_owned_cells,
                                                  other_index),
                                        get_index(locally_owned_cells, index));
                }
          }
      }
  }
} // namespace dealii::PolyUtils::internal


namespace dealii::PolyUtils
{
  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  struct Rtree_visitor : public boost::geometry::index::detail::rtree::visitor<
                           Value,
                           typename Options::parameters_type,
                           Box,
                           Allocators,
                           typename Options::node_tag,
                           true>::type
  {
    inline Rtree_visitor(
      const Translator &translator,
      unsigned int      target_level,
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>> &boxes,
      std::vector<std::vector<unsigned int>>                            &csr);


    /**
     * An alias that identifies an InternalNode of the tree.
     */
    using InternalNode =
      typename boost::geometry::index::detail::rtree::internal_node<
        Value,
        typename Options::parameters_type,
        Box,
        Allocators,
        typename Options::node_tag>::type;

    /**
     * An alias that identifies a Leaf of the tree.
     */
    using Leaf = typename boost::geometry::index::detail::rtree::leaf<
      Value,
      typename Options::parameters_type,
      Box,
      Allocators,
      typename Options::node_tag>::type;

    /**
     * Implements the visitor interface for InternalNode objects. If the node
     * belongs to the level next to @p target_level, then fill the bounding box vector for that node.
     */
    inline void
    operator()(const InternalNode &node);

    /**
     * Implements the visitor interface for Leaf objects.
     */
    inline void
    operator()(const Leaf &);

    /**
     * Translator interface, required by the boost implementation of the rtree.
     */
    const Translator &translator;

    /**
     * Store the level we are currently visiting.
     */
    size_t level;

    /**
     * Index used to keep track of the number of different visited nodes during
     * recursion/
     */
    size_t node_counter;

    size_t next_level_leafs_processed;
    /**
     * The level where children are living.
     * Before: "we want to extract from the RTree object."
     */
    const size_t target_level;

    /**
     * A reference to the input vector of vector of BoundingBox objects. This
     * vector v has the following property: v[i] = vector with all
     * of the BoundingBox bounded by the i-th node of the Rtree.
     */
    std::vector<std::vector<typename Triangulation<
      boost::geometry::dimension<Box>::value>::active_cell_iterator>>
      &agglomerates;

    std::vector<std::vector<unsigned int>> &row_ptr;
  };



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::Rtree_visitor(
    const Translator  &translator,
    const unsigned int target_level,
    std::vector<std::vector<typename Triangulation<
      boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                           &bb_in_boxes,
    std::vector<std::vector<unsigned int>> &csr)
    : translator(translator)
    , level(0)
    , node_counter(0)
    , next_level_leafs_processed(0)
    , target_level(target_level)
    , agglomerates(bb_in_boxes)
    , row_ptr(csr)
  {}



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  void
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
    const Rtree_visitor::InternalNode &node)
  {
    using elements_type =
      typename boost::geometry::index::detail::rtree::elements_type<
        InternalNode>::type; //  pairs of bounding box and pointer to child node
    const elements_type &elements =
      boost::geometry::index::detail::rtree::elements(node);

    if (level < target_level)
      {
        size_t level_backup = level;
        ++level;

        for (typename elements_type::const_iterator it = elements.begin();
             it != elements.end();
             ++it)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *it->second);
          }

        level = level_backup;
      }
    else if (level == target_level)
      {
        // const unsigned int n_children = elements.size();
        const auto offset = agglomerates.size();
        agglomerates.resize(offset + 1);
        row_ptr.resize(row_ptr.size() + 1);
        next_level_leafs_processed = 0;
        row_ptr.back().push_back(
          next_level_leafs_processed); // convention: row_ptr[0]=0
        size_t level_backup = level;

        ++level;
        for (const auto &child : elements)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *child.second);
          }
        // Done with node number 'node_counter'

        ++node_counter; // visited all children of an internal node

        level = level_backup;
      }
    else if (level > target_level)
      {
        // Keep visiting until you go to the leafs.
        size_t level_backup = level;

        ++level;

        for (const auto &child : elements)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *child.second);
          }
        level = level_backup;
        row_ptr[node_counter].push_back(next_level_leafs_processed);
      }
  }



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  void
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
    const Rtree_visitor::Leaf &leaf)
  {
    using elements_type =
      typename boost::geometry::index::detail::rtree::elements_type<
        Leaf>::type; //  pairs of bounding box and pointer to child node
    const elements_type &elements =
      boost::geometry::index::detail::rtree::elements(leaf);


    for (const auto &it : elements)
      {
        agglomerates[node_counter].push_back(it.second);
      }
    next_level_leafs_processed += elements.size();
  }

  template <typename Rtree>
  inline std::pair<
    std::vector<std::vector<unsigned int>>,
    std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
      typename Rtree::indexable_type>::value>::active_cell_iterator>>>
  extract_children_of_level(const Rtree &tree, const unsigned int level)
  {
    using RtreeView =
      boost::geometry::index::detail::rtree::utilities::view<Rtree>;
    RtreeView rtv(tree);

    std::vector<std::vector<unsigned int>> csrs;
    std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
      typename Rtree::indexable_type>::value>::active_cell_iterator>>
      agglomerates;

    if (rtv.depth() == 0)
      {
        // The below algorithm does not work for `rtv.depth()==0`, which might
        // happen if the number entries in the tree is too small.
        // In this case, simply return a single bounding box.
        agglomerates.resize(1);
        agglomerates[0].resize(1);
        csrs.resize(1);
        csrs[0].resize(1);
      }
    else
      {
        const unsigned int target_level =
          std::min<unsigned int>(level, rtv.depth());

        Rtree_visitor<typename RtreeView::value_type,
                      typename RtreeView::options_type,
                      typename RtreeView::translator_type,
                      typename RtreeView::box_type,
                      typename RtreeView::allocators_type>
          node_visitor(rtv.translator(), target_level, agglomerates, csrs);
        rtv.apply_visitor(node_visitor);
      }
    AssertDimension(agglomerates.size(), csrs.size());

    return {csrs, agglomerates};
  }


  template <int dim, typename Number = double>
  Number
  compute_h_orthogonal(
    const unsigned int face_index,
    const std::vector<typename Triangulation<dim>::active_face_iterator>
                         &polygon_boundary,
    const Tensor<1, dim> &deal_normal)
  {
#ifdef DEAL_II_WITH_CGAL

    using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
    std::vector<typename Kernel::FT> candidates;
    candidates.reserve(polygon_boundary.size() - 1);

    // Initialize the range of faces to be checked for intersection: they are
    // {0,..,n_faces-1}\setminus the current face index face_index.
    std::vector<unsigned int> face_indices(polygon_boundary.size());
    std::iota(face_indices.begin(), face_indices.end(), 0); // fill the range
    face_indices.erase(face_indices.cbegin() +
                       face_index); // remove current index

    if constexpr (dim == 2)
      {
        typename Kernel::Segment_2 face_segm(
          {polygon_boundary[face_index]->vertex(0)[0],
           polygon_boundary[face_index]->vertex(0)[1]},
          {polygon_boundary[face_index]->vertex(1)[0],
           polygon_boundary[face_index]->vertex(1)[1]});

        // Shoot a ray from the midpoint of the face in the orthogonal direction
        // given by deal.II normals
        const auto &midpoint = CGAL::midpoint(face_segm);
        // deal.II normal is always outward, flip the direction
        const typename Kernel::Vector_2 orthogonal_direction{-deal_normal[0],
                                                             -deal_normal[1]};
        const typename Kernel::Ray_2    ray(midpoint, orthogonal_direction);
        for (const auto f : face_indices)
          {
            typename Kernel::Segment_2 segm({polygon_boundary[f]->vertex(0)[0],
                                             polygon_boundary[f]->vertex(0)[1]},
                                            {polygon_boundary[f]->vertex(1)[0],
                                             polygon_boundary[f]->vertex(
                                               1)[1]});

            if (CGAL::do_intersect(ray, segm))
              candidates.push_back(CGAL::squared_distance(midpoint, segm));
          }
        return std::sqrt(CGAL::to_double(
          *std::min_element(candidates.cbegin(), candidates.cend())));
      }
    else if constexpr (dim == 3)
      {
        const typename Kernel::Point_3 &center{
          polygon_boundary[face_index]->center()[0],
          polygon_boundary[face_index]->center()[1],
          polygon_boundary[face_index]->center()[2]};
        // deal.II normal is always outward, flip the direction
        const typename Kernel::Vector_3 orthogonal_direction{-deal_normal[0],
                                                             -deal_normal[1],
                                                             -deal_normal[2]};
        const typename Kernel::Ray_3    ray(center, orthogonal_direction);

        for (const auto f : face_indices)
          {
            // split the face into 2 triangles and compute distances
            typename Kernel::Triangle_3 first_triangle(
              {polygon_boundary[f]->vertex(0)[0],
               polygon_boundary[f]->vertex(0)[1],
               polygon_boundary[f]->vertex(0)[2]},
              {polygon_boundary[f]->vertex(1)[0],
               polygon_boundary[f]->vertex(1)[1],
               polygon_boundary[f]->vertex(1)[2]},
              {polygon_boundary[f]->vertex(3)[0],
               polygon_boundary[f]->vertex(3)[1],
               polygon_boundary[f]->vertex(3)[2]});
            typename Kernel::Triangle_3 second_triangle(
              {polygon_boundary[f]->vertex(0)[0],
               polygon_boundary[f]->vertex(0)[1],
               polygon_boundary[f]->vertex(0)[2]},
              {polygon_boundary[f]->vertex(3)[0],
               polygon_boundary[f]->vertex(3)[1],
               polygon_boundary[f]->vertex(3)[2]},
              {polygon_boundary[f]->vertex(2)[0],
               polygon_boundary[f]->vertex(2)[1],
               polygon_boundary[f]->vertex(2)[2]});

            // compute point-triangle distance only if the orthogonal ray
            // hits the triangle
            if (CGAL::do_intersect(ray, first_triangle))
              candidates.push_back(
                CGAL::squared_distance(center, first_triangle));
            if (CGAL::do_intersect(ray, second_triangle))
              candidates.push_back(
                CGAL::squared_distance(center, second_triangle));
          }

        return std::sqrt(CGAL::to_double(
          *std::min_element(candidates.cbegin(), candidates.cend())));
      }
    else
      {
        Assert(false, ExcImpossibleInDim(dim));
        (void)face_index;
        (void)polygon_boundary;
        return {};
      }

#else

    Assert(false, ExcNeedsCGAL());
    (void)face_index;
    (void)polygon_boundary;
    return {};
#endif
  }



  /**
   * Agglomerate cells together based on their global index. This function is
   * **not** efficient and should be used for testing purposes only.
   */
  template <int dim, int spacedim = dim>
  void
  collect_cells_for_agglomeration(
    const Triangulation<dim, spacedim>          &tria,
    const std::vector<types::global_cell_index> &cell_idxs,
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
      &cells_to_be_agglomerated)
  {
    Assert(cells_to_be_agglomerated.size() == 0,
           ExcMessage(
             "The vector of cells is supposed to be filled by this function."));
    for (const auto &cell : tria.active_cell_iterators())
      if (std::find(cell_idxs.begin(),
                    cell_idxs.end(),
                    cell->active_cell_index()) != cell_idxs.end())
        {
          cells_to_be_agglomerated.push_back(cell);
        }
  }



  /**
   * Partition with METIS the locally owned regions of the given
   * triangulation.
   *
   * @note The given triangulation must be a parallel::fullydistributed::Triangulation. This is
   * required as the partitions generated by p4est, the partitioner for
   * parallell::distributed::Triangulation, can generate discontinuous
   * partitions which are not supported by the METIS partitioner.
   *
   */
  template <int dim, int spacedim>
  void
  partition_locally_owned_regions(const unsigned int            n_partitions,
                                  Triangulation<dim, spacedim> &triangulation,
                                  const SparsityTools::Partitioner partitioner)
  {
    AssertDimension(dim, spacedim);
    Assert(n_partitions > 0,
           ExcMessage("Invalid number of partitions, you provided " +
                      std::to_string(n_partitions)));

    auto parallel_triangulation =
      dynamic_cast<parallel::fullydistributed::Triangulation<dim, spacedim> *>(
        &triangulation);
    Assert(
      (parallel_triangulation != nullptr),
      ExcMessage(
        "Only fully distributed triangulations are supported. If you are using"
        "a parallel::distributed::triangulation, you must convert it to a fully"
        "distributed as explained in the documentation."));

    // check for an easy return
    if (n_partitions == 1)
      {
        for (const auto &cell : parallel_triangulation->active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_material_id(0);
        return;
      }

    // collect all locally owned cells
    std::vector<types::global_cell_index> locally_owned_cells;
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        locally_owned_cells.push_back(cell->active_cell_index());

    DynamicSparsityPattern cell_connectivity;
    internal::get_face_connectivity_of_cells(*parallel_triangulation,
                                             cell_connectivity,
                                             locally_owned_cells);

    SparsityPattern sp_cell_connectivity;
    sp_cell_connectivity.copy_from(cell_connectivity);

    // partition each locally owned connection graph and get
    // back a vector of indices, one per degree
    // of freedom (which is associated with a
    // cell)
    std::vector<unsigned int> partition_indices(
      parallel_triangulation->n_locally_owned_active_cells());
    SparsityTools::partition(sp_cell_connectivity,
                             n_partitions,
                             partition_indices,
                             partitioner);


    // finally loop over all cells and set the material ids
    for (const auto &cell : parallel_triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_material_id(
          partition_indices[internal::get_index(locally_owned_cells,
                                                cell->active_cell_index())]);
  }



  /**
   * Partition with METIS the locally owned regions of the given
   * triangulation and insert agglomerates in the polytopic grid.
   *
   * @note The given triangulation must be a parallel::fullydistributed::Triangulation. This is
   * required as the partitions generated by p4est, the partitioner for
   * parallell::distributed::Triangulation, can generate discontinuous
   * partitions which are not supported by the METIS partitioner.
   *
   */
  template <int dim, int spacedim>
  void
  partition_locally_owned_regions(
    AgglomerationHandler<dim>       &agglomeration_handler,
    const unsigned int               n_partitions,
    Triangulation<dim, spacedim>    &triangulation,
    const SparsityTools::Partitioner partitioner)
  {
    AssertDimension(dim, spacedim);
    Assert(
      agglomeration_handler.n_agglomerates() == 0,
      ExcMessage(
        "The agglomerated grid must be empty upon calling this function."));
    Assert(n_partitions > 0,
           ExcMessage("Invalid number of partitions, you provided " +
                      std::to_string(n_partitions)));

    auto parallel_triangulation =
      dynamic_cast<parallel::fullydistributed::Triangulation<dim, spacedim> *>(
        &triangulation);
    Assert(
      (parallel_triangulation != nullptr),
      ExcMessage(
        "Only fully distributed triangulations are supported. If you are using"
        "a parallel::distributed::triangulation, you must convert it to a"
        "fully distributed as explained in the documentation."));

    // check for an easy return
    if (n_partitions == 1)
      {
        for (const auto &cell : parallel_triangulation->active_cell_iterators())
          if (cell->is_locally_owned())
            agglomeration_handler.define_agglomerate({cell});
        return;
      }

    // collect all locally owned cells
    std::vector<types::global_cell_index> locally_owned_cells;
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        locally_owned_cells.push_back(cell->active_cell_index());

    DynamicSparsityPattern cell_connectivity;
    internal::get_face_connectivity_of_cells(*parallel_triangulation,
                                             cell_connectivity,
                                             locally_owned_cells);

    SparsityPattern sp_cell_connectivity;
    sp_cell_connectivity.copy_from(cell_connectivity);

    // partition each locally owned connection graph and get
    // back a vector of indices, one per degree
    // of freedom (which is associated with a
    // cell)
    std::vector<unsigned int> partition_indices(
      parallel_triangulation->n_locally_owned_active_cells());
    SparsityTools::partition(sp_cell_connectivity,
                             n_partitions,
                             partition_indices,
                             partitioner);

    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      cells_per_partion_id;
    cells_per_partion_id.resize(n_partitions); // number of agglomerates

    // finally loop over all cells and store the ones with same partition index
    for (const auto &cell : parallel_triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
        cells_per_partion_id[partition_indices[internal::get_index(
                               locally_owned_cells, cell->active_cell_index())]]
          .push_back(cell);

    // All the cells with the same partition index will be merged together.
    for (unsigned int i = 0; i < n_partitions; ++i)
      agglomeration_handler.define_agglomerate(cells_per_partion_id[i]);
  }



  template <int dim>
  std::
    tuple<std::vector<double>, std::vector<double>, std::vector<double>, double>
    compute_quality_metrics(const AgglomerationHandler<dim> &ah)
  {
    static_assert(dim == 2); // only 2D case is implemented.
#ifdef DEAL_II_WITH_CGAL
    using Kernel = CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt;
    using Polygon_with_holes = typename CGAL::Polygon_with_holes_2<Kernel>;
    using Gt    = typename CGAL::Segment_Delaunay_graph_traits_2<Kernel>;
    using SDG2  = typename CGAL::Segment_Delaunay_graph_2<Gt>;
    using CDT   = typename CGAL::Constrained_Delaunay_triangulation_2<Kernel>;
    using CDTP  = typename CGAL::Constrained_triangulation_plus_2<CDT>;
    using Point = typename CDTP::Point;
    using Cid   = typename CDTP::Constraint_id;
    using Vertex_handle = typename CDTP::Vertex_handle;


    const auto compute_radius_inscribed_circle =
      [](const CGAL::Polygon_2<Kernel> &polygon) -> double {
      SDG2 sdg;

      sdg.insert_segments(polygon.edges_begin(), polygon.edges_end());

      double                               sd = 0, sqdist = 0;
      typename SDG2::Finite_faces_iterator fit = sdg.finite_faces_begin();
      for (; fit != sdg.finite_faces_end(); ++fit)
        {
          typename Kernel::Point_2 pp = sdg.primal(fit);
          for (int i = 0; i < 3; ++i)
            {
              assert(!sdg.is_infinite(fit->vertex(i)));
              if (fit->vertex(i)->site().is_segment())
                {
                  typename Kernel::Segment_2 s =
                    fit->vertex(i)->site().segment();
                  sqdist = CGAL::to_double(CGAL::squared_distance(pp, s));
                }
              else
                {
                  typename Kernel::Point_2 p = fit->vertex(i)->site().point();
                  sqdist = CGAL::to_double(CGAL::squared_distance(pp, p));
                }
            }

          if (polygon.bounded_side(pp) == CGAL::ON_BOUNDED_SIDE)
            sd = std::max(sqdist, sd);
        }

      return std::sqrt(sd);
    };

    const auto mesh_size = [&ah]() -> double {
      double hmax = 0.;
      for (const auto &polytope : ah.polytope_iterators())
        if (polytope->is_locally_owned())
          {
            const double diameter = polytope->diameter();
            if (diameter > hmax)
              hmax = diameter;
          }
      return hmax;
    }();


    // vectors holding quality metrics

    // ration between radius of radius_inscribed_circle and circumscribed circle
    std::vector<double> circle_ratios;
    std::vector<double> unformity_factors; // diameter of element over mesh size
    std::vector<double>
      box_ratio; // ratio between measure of bbox and measure of element.

    const std::vector<BoundingBox<dim>> &bboxes = ah.get_local_bboxes();
    // Loop over all polytopes and compute metrics.
    for (const auto &polytope : ah.polytope_iterators())
      {
        if (polytope->is_locally_owned())
          {
            const std::vector<typename Triangulation<dim>::active_face_iterator>
              &boundary = polytope->polytope_boundary();

            const double diameter                    = polytope->diameter();
            const double radius_circumscribed_circle = .5 * diameter;

            CDTP cdtp;
            for (unsigned int f = 0; f < boundary.size(); f += 1)
              {
                // polyline
                cdtp.insert_constraint(
                  {boundary[f]->vertex(0)[0], boundary[f]->vertex(0)[1]},
                  {boundary[f]->vertex(1)[0], boundary[f]->vertex(1)[1]});
              }
            cdtp.split_subconstraint_graph_into_constraints();

            CGAL::Polygon_2<Kernel> outer_polygon;
            auto                    it = outer_polygon.vertices_begin();
            for (typename CDTP::Constraint_id cid : cdtp.constraints())
              {
                for (typename CDTP::Vertex_handle vh :
                     cdtp.vertices_in_constraint(cid))
                  {
                    it = outer_polygon.insert(outer_polygon.vertices_end(),
                                              vh->point());
                  }
              }
            outer_polygon.erase(it); // remove duplicate final point

            const double radius_inscribed_circle =
              compute_radius_inscribed_circle(outer_polygon);

            circle_ratios.push_back(radius_inscribed_circle /
                                    radius_circumscribed_circle);
            unformity_factors.push_back(diameter / mesh_size);

            // box_ratio

            const auto  &agglo_values = ah.reinit(polytope);
            const double measure_element =
              std::accumulate(agglo_values.get_JxW_values().cbegin(),
                              agglo_values.get_JxW_values().cend(),
                              0.);
            box_ratio.push_back(measure_element /
                                bboxes[polytope->index()].volume());
          }
      }



    // Get all of the local bounding boxes
    double covering_bboxes = 0.;
    for (unsigned int i = 0; i < bboxes.size(); ++i)
      covering_bboxes += bboxes[i].volume();

    const double overlap_factor =
      Utilities::MPI::sum(covering_bboxes,
                          ah.get_dof_handler().get_mpi_communicator()) /
      GridTools::volume(ah.get_triangulation()); // assuming a linear mapping



    return {unformity_factors, circle_ratios, box_ratio, overlap_factor};
#else

    (void)ah;
    return {};
#endif
  }


  /**
   * Export each polygon in a csv file as a collection of segments.
   */
  template <int dim>
  void
  export_polygon_to_csv_file(
    const AgglomerationHandler<dim> &agglomeration_handler,
    const std::string               &filename)
  {
    static_assert(dim == 2); // With 3D, Paraview is much better
    std::ofstream myfile;
    myfile.open(filename + ".csv");

    for (const auto &polytope : agglomeration_handler.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          const std::vector<typename Triangulation<dim>::active_face_iterator>
            &boundary = polytope->polytope_boundary();
          for (unsigned int f = 0; f < boundary.size(); ++f)
            {
              myfile << boundary[f]->vertex(0)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(0)[1];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[1];
              myfile << "\n";
            }
        }


    myfile.close();
  }


  template <typename T>
  inline constexpr T
  constexpr_pow(T num, unsigned int pow)
  {
    return (pow >= sizeof(unsigned int) * 8) ? 0 :
           pow == 0                          ? 1 :
                                               num * constexpr_pow(num, pow - 1);
  }



  void
  write_to_matrix_market_format(const std::string &filename,
                                const std::string &matrix_name,
                                const TrilinosWrappers::SparseMatrix &matrix)
  {
#ifdef DEAL_II_WITH_TRILINOS
    const Epetra_CrsMatrix &trilinos_matrix = matrix.trilinos_matrix();

    const int ierr =
      EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),
                                             trilinos_matrix,
                                             matrix_name.c_str(),
                                             0 /*description field empty*/,
                                             true /*write header*/);
    AssertThrow(ierr == 0, ExcTrilinosError(ierr));
#else
    (void)filename;
    (void)matrix_name;
    (void)matrix;
#endif
  }



  namespace internal
  {
    /**
     * Same as the public free function with the same name, but storing
     * explicitly the interpolation matrix and performing interpolation through
     * matrix-vector product.
     */
    template <int dim, int spacedim, typename VectorType>
    void
    interpolate_to_fine_grid(
      const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
      VectorType                                &dst,
      const VectorType                          &src)
    {
      Assert((dim == spacedim), ExcNotImplemented());
      Assert(
        dst.size() == 0,
        ExcMessage(
          "The destination vector must the empt upon calling this function."));

      using NumberType = typename VectorType::value_type;
      constexpr bool is_trilinos_vector =
        std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;
      using MatrixType = std::conditional_t<is_trilinos_vector,
                                            TrilinosWrappers::SparseMatrix,
                                            SparseMatrix<NumberType>>;

      MatrixType interpolation_matrix;

      [[maybe_unused]]
      typename std::conditional_t<!is_trilinos_vector, SparsityPattern, void *>
        sp;

      // Get some info from the handler
      const DoFHandler<dim, spacedim> &agglo_dh =
        agglomeration_handler.agglo_dh;

      DoFHandler<dim, spacedim> *output_dh =
        const_cast<DoFHandler<dim, spacedim> *>(
          &agglomeration_handler.output_dh);
      const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
      const Mapping<dim> &mapping = agglomeration_handler.get_mapping();
      const Triangulation<dim, spacedim> &tria =
        agglomeration_handler.get_triangulation();
      const auto &bboxes = agglomeration_handler.get_local_bboxes();

      std::unique_ptr<FiniteElement<dim>> output_fe;
      if (tria.all_reference_cells_are_hyper_cube())
        output_fe = std::make_unique<FE_DGQ<dim>>(fe.degree);
      else if (tria.all_reference_cells_are_simplex())
        output_fe = std::make_unique<FE_SimplexDGP<dim>>(fe.degree);
      else
        AssertThrow(false, ExcNotImplemented());

      // Setup an auxiliary DoFHandler for output purposes
      output_dh->reinit(tria);
      output_dh->distribute_dofs(*output_fe);

      const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
      const IndexSet  locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(*output_dh);

      const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


      DynamicSparsityPattern dsp(output_dh->n_dofs(),
                                 agglo_dh.n_dofs(),
                                 locally_relevant_dofs);

      std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
      std::vector<types::global_dof_index> standard_dof_indices(
        fe.dofs_per_cell);
      std::vector<types::global_dof_index> output_dof_indices(
        output_fe->dofs_per_cell);

      Quadrature<dim>         quad(output_fe->get_unit_support_points());
      FEValues<dim, spacedim> output_fe_values(mapping,
                                               *output_fe,
                                               quad,
                                               update_quadrature_points);

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                for (const auto &slave : slaves)
                  {
                    // addd master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);
                    slave_output->get_dof_indices(output_dof_indices);
                    for (const auto row : output_dof_indices)
                      dsp.add_entries(row,
                                      agglo_dof_indices.begin(),
                                      agglo_dof_indices.end());
                  }
              }
          }


      const auto assemble_interpolation_matrix = [&]() {
        FullMatrix<NumberType> local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
        std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

        // Dummy AffineConstraints, only needed for loc2glb
        AffineConstraints<NumberType> c;
        c.close();

        for (const auto &cell : agglo_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              if (agglomeration_handler.is_master_cell(cell))
                {
                  auto slaves = agglomeration_handler.get_slaves_of_idx(
                    cell->active_cell_index());
                  slaves.emplace_back(cell);

                  cell->get_dof_indices(agglo_dof_indices);

                  const types::global_cell_index polytope_index =
                    agglomeration_handler.cell_to_polytope_index(cell);

                  // Get the box of this agglomerate.
                  const BoundingBox<dim> &box = bboxes[polytope_index];

                  for (const auto &slave : slaves)
                    {
                      // add master-slave relationship
                      const auto slave_output =
                        slave->as_dof_handler_iterator(*output_dh);

                      slave_output->get_dof_indices(output_dof_indices);
                      output_fe_values.reinit(slave_output);

                      local_matrix = 0.;

                      const auto &q_points =
                        output_fe_values.get_quadrature_points();
                      for (const auto i : output_fe_values.dof_indices())
                        {
                          const auto &p = box.real_to_unit(q_points[i]);
                          for (const auto j : output_fe_values.dof_indices())
                            {
                              local_matrix(i, j) = fe.shape_value(j, p);
                            }
                        }
                      c.distribute_local_to_global(local_matrix,
                                                   output_dof_indices,
                                                   agglo_dof_indices,
                                                   interpolation_matrix);
                    }
                }
            }
      };


      if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
        {
          const MPI_Comm &communicator = tria.get_mpi_communicator();
          SparsityTools::distribute_sparsity_pattern(dsp,
                                                     locally_owned_dofs,
                                                     communicator,
                                                     locally_relevant_dofs);

          interpolation_matrix.reinit(locally_owned_dofs,
                                      locally_owned_dofs_agglo,
                                      dsp,
                                      communicator);
          dst.reinit(locally_owned_dofs);
          assemble_interpolation_matrix();
        }
      else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
        {
          sp.copy_from(dsp);
          interpolation_matrix.reinit(sp);
          dst.reinit(output_dh->n_dofs());
          assemble_interpolation_matrix();
        }
      else
        {
          // PETSc, LA::d::v options not implemented.
          (void)agglomeration_handler;
          (void)dst;
          (void)src;
          AssertThrow(false, ExcNotImplemented());
        }

      // If tria is distributed
      if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &tria) != nullptr)
        interpolation_matrix.compress(VectorOperation::add);

      // Finally, perform the interpolation.
      interpolation_matrix.vmult(dst, src);
    }
  } // namespace internal



  /**
   * Given a vector @p src, typically the solution stemming after the
   * agglomerate problem has been solved, this function interpolates @p src
   * onto the finer grid and stores the result in vector @p dst. The last
   * argument @p on_the_fly does not build any interpolation matrix and allows
   * computing the entries in @p dst in a matrix-free fashion.
   *
   * @note Supported parallel types are TrilinosWrappers::SparseMatrix and
   * TrilinosWrappers::MPI::Vector.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  interpolate_to_fine_grid(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    VectorType                                &dst,
    const VectorType                          &src,
    const bool                                 on_the_fly = true)
  {
    Assert((dim == spacedim), ExcNotImplemented());
    Assert(
      dst.size() == 0,
      ExcMessage(
        "The destination vector must the empt upon calling this function."));

    using NumberType = typename VectorType::value_type;
    static constexpr bool is_trilinos_vector =
      std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;

    static constexpr bool is_supported_vector =
      std::is_same_v<VectorType, Vector<NumberType>> || is_trilinos_vector;
    static_assert(is_supported_vector);

    // First, check for an easy return
    if (on_the_fly == false)
      {
        return internal::interpolate_to_fine_grid(agglomeration_handler,
                                                  dst,
                                                  src);
      }
    else
      {
        // otherwise, do not create any matrix
        if (!agglomeration_handler.used_fe_collection())
          {
            // Original version: handle case without hp::FECollection
            const Triangulation<dim, spacedim> &tria =
              agglomeration_handler.get_triangulation();
            const Mapping<dim> &mapping = agglomeration_handler.get_mapping();
            const FiniteElement<dim, spacedim> &original_fe =
              agglomeration_handler.get_fe();

            // We use DGQ (on tensor-product meshes) or DGP (on simplex meshes)
            // nodal elements of the same degree as the ones in the
            // agglomeration handler to interpolate the solution onto the finer
            // grid.
            std::unique_ptr<FiniteElement<dim>> output_fe;
            if (tria.all_reference_cells_are_hyper_cube())
              output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);
            else if (tria.all_reference_cells_are_simplex())
              output_fe =
                std::make_unique<FE_SimplexDGP<dim>>(original_fe.degree);
            else
              AssertThrow(false, ExcNotImplemented());

            DoFHandler<dim> &output_dh =
              const_cast<DoFHandler<dim> &>(agglomeration_handler.output_dh);
            output_dh.reinit(tria);
            output_dh.distribute_dofs(*output_fe);

            if constexpr (std::is_same_v<VectorType,
                                         TrilinosWrappers::MPI::Vector>)
              {
                const IndexSet &locally_owned_dofs =
                  output_dh.locally_owned_dofs();
                dst.reinit(locally_owned_dofs);
              }
            else if constexpr (std::is_same_v<VectorType, Vector<NumberType>>)
              {
                dst.reinit(output_dh.n_dofs());
              }
            else
              {
                // PETSc, LA::d::v options not implemented.
                (void)agglomeration_handler;
                (void)dst;
                (void)src;
                AssertThrow(false, ExcNotImplemented());
              }



            const unsigned int dofs_per_cell =
              agglomeration_handler.n_dofs_per_cell();
            const unsigned int output_dofs_per_cell =
              output_fe->n_dofs_per_cell();
            Quadrature<dim> quad(output_fe->get_unit_support_points());
            FEValues<dim>   output_fe_values(mapping,
                                           *output_fe,
                                           quad,
                                           update_quadrature_points);

            std::vector<types::global_dof_index> local_dof_indices(
              dofs_per_cell);
            std::vector<types::global_dof_index> local_dof_indices_output(
              output_dofs_per_cell);

            const auto &bboxes = agglomeration_handler.get_local_bboxes();
            for (const auto &polytope :
                 agglomeration_handler.polytope_iterators())
              {
                if (polytope->is_locally_owned())
                  {
                    polytope->get_dof_indices(local_dof_indices);
                    const BoundingBox<dim> &box = bboxes[polytope->index()];

                    const auto &deal_cells =
                      polytope->get_agglomerate(); // fine deal.II cells
                    for (const auto &cell : deal_cells)
                      {
                        const auto slave_output = cell->as_dof_handler_iterator(
                          agglomeration_handler.output_dh);
                        slave_output->get_dof_indices(local_dof_indices_output);
                        output_fe_values.reinit(slave_output);

                        const auto &qpoints =
                          output_fe_values.get_quadrature_points();

                        for (unsigned int j = 0; j < output_dofs_per_cell; ++j)
                          {
                            const auto &ref_qpoint =
                              box.real_to_unit(qpoints[j]);
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              dst(local_dof_indices_output[j]) +=
                                src(local_dof_indices[i]) *
                                original_fe.shape_value(i, ref_qpoint);
                          }
                      }
                  }
              }
          }
        else
          {
            // Handle the hp::FECollection case
            const Triangulation<dim, spacedim> &tria =
              agglomeration_handler.get_triangulation();
            const Mapping<dim> &mapping = agglomeration_handler.get_mapping();
            const hp::FECollection<dim, spacedim> &original_fe_collection =
              agglomeration_handler.get_fe_collection();

            // We use DGQ (on tensor-product meshes) or DGP (on simplex meshes)
            // nodal elements of the same degree as the ones in the
            // agglomeration handler to interpolate the solution onto the finer
            // grid.
            hp::FECollection<dim, spacedim> output_fe_collection;

            Assert(original_fe_collection[0].n_components() >= 1,
                   ExcMessage("Invalid FE: must have at least one component."));
            if (original_fe_collection[0].n_components() == 1)
              {
                // Scalar case
                for (unsigned int i = 0; i < original_fe_collection.size(); ++i)
                  {
                    std::unique_ptr<FiniteElement<dim>> output_fe;
                    if (tria.all_reference_cells_are_hyper_cube())
                      output_fe = std::make_unique<FE_DGQ<dim>>(
                        original_fe_collection[i].degree);
                    else if (tria.all_reference_cells_are_simplex())
                      output_fe = std::make_unique<FE_SimplexDGP<dim>>(
                        original_fe_collection[i].degree);
                    else
                      AssertThrow(false, ExcNotImplemented());
                    output_fe_collection.push_back(*output_fe);
                  }
              }
            else if (original_fe_collection[0].n_components() > 1)
              {
                // System case
                for (unsigned int i = 0; i < original_fe_collection.size(); ++i)
                  {
                    std::vector<const FiniteElement<dim, spacedim> *>
                                              base_elements;
                    std::vector<unsigned int> multiplicities;
                    for (unsigned int b = 0;
                         b < original_fe_collection[i].n_base_elements();
                         ++b)
                      {
                        if (dynamic_cast<const FE_Nothing<dim> *>(
                              &original_fe_collection[i].base_element(b)))
                          base_elements.push_back(
                            new FE_Nothing<dim, spacedim>());
                        else
                          {
                            if (tria.all_reference_cells_are_hyper_cube())
                              base_elements.push_back(new FE_DGQ<dim, spacedim>(
                                original_fe_collection[i]
                                  .base_element(b)
                                  .degree));
                            else if (tria.all_reference_cells_are_simplex())
                              base_elements.push_back(
                                new FE_SimplexDGP<dim, spacedim>(
                                  original_fe_collection[i]
                                    .base_element(b)
                                    .degree));
                            else
                              AssertThrow(false, ExcNotImplemented());
                          }
                        multiplicities.push_back(
                          original_fe_collection[i].element_multiplicity(b));
                      }

                    FESystem<dim, spacedim> output_fe_system(base_elements,
                                                             multiplicities);
                    for (const auto *ptr : base_elements)
                      delete ptr;
                    output_fe_collection.push_back(output_fe_system);
                  }
              }


            DoFHandler<dim> &output_dh =
              const_cast<DoFHandler<dim> &>(agglomeration_handler.output_dh);
            output_dh.reinit(tria);
            for (const auto &polytope :
                 agglomeration_handler.polytope_iterators())
              {
                if (polytope->is_locally_owned())
                  {
                    const auto &deal_cells =
                      polytope->get_agglomerate(); // fine deal.II cells
                    const unsigned int active_fe_idx =
                      polytope->active_fe_index();

                    for (const auto &cell : deal_cells)
                      {
                        const typename DoFHandler<dim>::active_cell_iterator
                          slave_cell_dh_iterator =
                            cell->as_dof_handler_iterator(output_dh);
                        slave_cell_dh_iterator->set_active_fe_index(
                          active_fe_idx);
                      }
                  }
              }
            output_dh.distribute_dofs(output_fe_collection);

            if constexpr (std::is_same_v<VectorType,
                                         TrilinosWrappers::MPI::Vector>)
              {
                const IndexSet &locally_owned_dofs =
                  output_dh.locally_owned_dofs();
                dst.reinit(locally_owned_dofs);
              }
            else if constexpr (std::is_same_v<VectorType, Vector<NumberType>>)
              {
                dst.reinit(output_dh.n_dofs());
              }
            else
              {
                // PETSc, LA::d::v options not implemented.
                (void)agglomeration_handler;
                (void)dst;
                (void)src;
                AssertThrow(false, ExcNotImplemented());
              }

            const auto &bboxes = agglomeration_handler.get_local_bboxes();
            for (const auto &polytope :
                 agglomeration_handler.polytope_iterators())
              {
                if (polytope->is_locally_owned())
                  {
                    const unsigned int active_fe_idx =
                      polytope->active_fe_index();
                    const unsigned int dofs_per_cell =
                      polytope->get_fe().dofs_per_cell;
                    const unsigned int output_dofs_per_cell =
                      output_fe_collection[active_fe_idx].n_dofs_per_cell();
                    Quadrature<dim> quad(output_fe_collection[active_fe_idx]
                                           .get_unit_support_points());
                    FEValues<dim>   output_fe_values(
                      mapping,
                      output_fe_collection[active_fe_idx],
                      quad,
                      update_quadrature_points);
                    std::vector<types::global_dof_index> local_dof_indices(
                      dofs_per_cell);
                    std::vector<types::global_dof_index>
                      local_dof_indices_output(output_dofs_per_cell);

                    polytope->get_dof_indices(local_dof_indices);
                    const BoundingBox<dim> &box = bboxes[polytope->index()];

                    const auto &deal_cells =
                      polytope->get_agglomerate(); // fine deal.II cells
                    for (const auto &cell : deal_cells)
                      {
                        const auto slave_output = cell->as_dof_handler_iterator(
                          agglomeration_handler.output_dh);
                        slave_output->get_dof_indices(local_dof_indices_output);
                        output_fe_values.reinit(slave_output);

                        const auto &qpoints =
                          output_fe_values.get_quadrature_points();

                        for (unsigned int j = 0; j < output_dofs_per_cell; ++j)
                          {
                            const unsigned int component_idx_of_this_dof =
                              slave_output->get_fe()
                                .system_to_component_index(j)
                                .first;
                            const auto &ref_qpoint =
                              box.real_to_unit(qpoints[j]);
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              dst(local_dof_indices_output[j]) +=
                                src(local_dof_indices[i]) *
                                original_fe_collection[active_fe_idx]
                                  .shape_value_component(
                                    i, ref_qpoint, component_idx_of_this_dof);
                          }
                      }
                  }
              }
          }
      }
  }



  /**
   * Construct the interpolation matrix from the DG space defined the
   * polytopic elements defined in @p agglomeration_handler to the DG space
   * defined on the @p DoFHandler associated to standard shapes. The
   * interpolation matrix is assumed to be default-constructed and is filled
   * inside this function.
   */
  template <int dim, int spacedim, typename MatrixType>
  void
  fill_interpolation_matrix(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    MatrixType                                &interpolation_matrix)
  {
    Assert((dim == spacedim), ExcNotImplemented());

    using NumberType = typename MatrixType::value_type;
    constexpr bool is_trilinos_matrix =
      std::is_same_v<MatrixType, TrilinosWrappers::MPI::Vector>;

    [[maybe_unused]]
    typename std::conditional_t<!is_trilinos_matrix, SparsityPattern, void *>
      sp;

    // Get some info from the handler
    const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

    DoFHandler<dim, spacedim> *output_dh =
      const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
    const Mapping<dim, spacedim> &mapping = agglomeration_handler.get_mapping();
    const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
    const Triangulation<dim, spacedim> &tria =
      agglomeration_handler.get_triangulation();
    const auto &bboxes = agglomeration_handler.get_local_bboxes();

    // Setup an auxiliary DoFHandler for output purposes
    output_dh->reinit(tria);
    output_dh->distribute_dofs(fe);

    const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
    const IndexSet  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(*output_dh);

    const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


    DynamicSparsityPattern dsp(output_dh->n_dofs(),
                               agglo_dh.n_dofs(),
                               locally_relevant_dofs);

    std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

    Quadrature<dim>         quad(fe.get_unit_support_points());
    FEValues<dim, spacedim> output_fe_values(mapping,
                                             fe,
                                             quad,
                                             update_quadrature_points);

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              for (const auto &slave : slaves)
                {
                  // addd master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);
                  slave_output->get_dof_indices(output_dof_indices);
                  for (const auto row : output_dof_indices)
                    dsp.add_entries(row,
                                    agglo_dof_indices.begin(),
                                    agglo_dof_indices.end());
                }
            }
        }


    const auto assemble_interpolation_matrix = [&]() {
      FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
      std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

      // Dummy AffineConstraints, only needed for loc2glb
      AffineConstraints<NumberType> c;
      c.close();

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                const types::global_cell_index polytope_index =
                  agglomeration_handler.cell_to_polytope_index(cell);

                // Get the box of this agglomerate.
                const BoundingBox<dim> &box = bboxes[polytope_index];

                for (const auto &slave : slaves)
                  {
                    // add master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);

                    slave_output->get_dof_indices(output_dof_indices);
                    output_fe_values.reinit(slave_output);

                    local_matrix = 0.;

                    const auto &q_points =
                      output_fe_values.get_quadrature_points();
                    for (const auto i : output_fe_values.dof_indices())
                      {
                        const auto &p = box.real_to_unit(q_points[i]);
                        for (const auto j : output_fe_values.dof_indices())
                          {
                            local_matrix(i, j) = fe.shape_value(j, p);
                          }
                      }
                    c.distribute_local_to_global(local_matrix,
                                                 output_dof_indices,
                                                 agglo_dof_indices,
                                                 interpolation_matrix);
                  }
              }
          }
    };


    if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
      {
        const MPI_Comm &communicator = tria.get_mpi_communicator();
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   communicator,
                                                   locally_relevant_dofs);

        interpolation_matrix.reinit(locally_owned_dofs,
                                    locally_owned_dofs_agglo,
                                    dsp,
                                    communicator);
        assemble_interpolation_matrix();
      }
    else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
      {
        sp.copy_from(dsp);
        interpolation_matrix.reinit(sp);
        assemble_interpolation_matrix();
      }
    else
      {
        // PETSc, LA::d::v options not implemented.
        (void)agglomeration_handler;
        AssertThrow(false, ExcNotImplemented());
      }

    // If tria is distributed
    if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
          &tria) != nullptr)
      interpolation_matrix.compress(VectorOperation::add);
  }



  /**
   * Similar to VectorTools::compute_global_error(), but customized for
   * polytopic elements. Aside from the solution vector and a reference
   * function, this function takes in addition a vector @p norms with types
   * VectorTools::NormType to be computed and later stored in the last
   * argument @p global_errors.
   * In case of a parallel vector, the local errors are collected over each
   * processor and later a classical reduction operation is performed.
   */
  template <int dim, typename Number, typename VectorType>
  void
  compute_global_error(const AgglomerationHandler<dim> &agglomeration_handler,
                       const VectorType                &solution,
                       const Function<dim, Number>     &exact_solution,
                       const std::vector<VectorTools::NormType> &norms,
                       std::vector<double>                      &global_errors)
  {
    Assert(solution.size() > 0,
           ExcNotImplemented(
             "Solution vector must be non-empty upon calling this function."));
    Assert(std::any_of(norms.cbegin(),
                       norms.cend(),
                       [](VectorTools::NormType norm_type) {
                         return (norm_type ==
                                   VectorTools::NormType::H1_seminorm ||
                                 norm_type == VectorTools::NormType::L2_norm);
                       }),
           ExcMessage("Norm type not supported"));
    global_errors.resize(norms.size());
    std::fill(global_errors.begin(), global_errors.end(), 0.);

    // Vector storing errors local to the current processor.
    std::vector<double> local_errors(norms.size());
    std::fill(local_errors.begin(), local_errors.end(), 0.);

    // Get some info from the handler
    const unsigned int dofs_per_cell = agglomeration_handler.n_dofs_per_cell();

    const bool compute_semi_H1 =
      std::any_of(norms.cbegin(),
                  norms.cend(),
                  [](VectorTools::NormType norm_type) {
                    return norm_type == VectorTools::NormType::H1_seminorm;
                  });

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &polytope : agglomeration_handler.polytope_iterators())
      {
        if (polytope->is_locally_owned())
          {
            const auto &agglo_values = agglomeration_handler.reinit(polytope);
            polytope->get_dof_indices(local_dof_indices);

            const auto         &q_points = agglo_values.get_quadrature_points();
            const unsigned int  n_qpoints = q_points.size();
            std::vector<double> analyical_sol_at_qpoints(n_qpoints);
            exact_solution.value_list(q_points, analyical_sol_at_qpoints);
            std::vector<Tensor<1, dim>> grad_analyical_sol_at_qpoints(
              n_qpoints);

            if (compute_semi_H1)
              exact_solution.gradient_list(q_points,
                                           grad_analyical_sol_at_qpoints);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                double         solution_at_qpoint = 0.;
                Tensor<1, dim> grad_solution_at_qpoint;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    solution_at_qpoint += solution(local_dof_indices[i]) *
                                          agglo_values.shape_value(i, q_index);

                    if (compute_semi_H1)
                      grad_solution_at_qpoint +=
                        solution(local_dof_indices[i]) *
                        agglo_values.shape_grad(i, q_index);
                  }
                // L2
                local_errors[0] += std::pow((analyical_sol_at_qpoints[q_index] -
                                             solution_at_qpoint),
                                            2) *
                                   agglo_values.JxW(q_index);

                // H1 seminorm
                if (compute_semi_H1)
                  for (unsigned int d = 0; d < dim; ++d)
                    local_errors[1] +=
                      std::pow((grad_analyical_sol_at_qpoints[q_index][d] -
                                grad_solution_at_qpoint[d]),
                               2) *
                      agglo_values.JxW(q_index);
              }
          }
      }

    // Perform reduction and take sqrt of each error
    global_errors[0] = Utilities::MPI::reduce<double>(
      local_errors[0],
      agglomeration_handler.get_triangulation().get_mpi_communicator(),
      [](const double a, const double b) { return a + b; });

    global_errors[0] = std::sqrt(global_errors[0]);

    if (compute_semi_H1)
      {
        global_errors[1] = Utilities::MPI::reduce<double>(
          local_errors[1],
          agglomeration_handler.get_triangulation().get_mpi_communicator(),
          [](const double a, const double b) { return a + b; });
        global_errors[1] = std::sqrt(global_errors[1]);
      }
  }



  /**
   * Utility function that builds the multilevel hierarchy from the tree level
   * @p starting_level. This function fills the vector of
   * @p AgglomerationHandlers objects by distributing degrees of freedom on
   * each level of the hierarchy. It returns the total number of levels in the
   * hierarchy.
   */
  template <int dim>
  unsigned int
  construct_agglomerated_levels(
    const Triangulation<dim> &tria,
    std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
                       &agglomeration_handlers,
    const FE_DGQ<dim>  &fe_dg,
    const Mapping<dim> &mapping,
    const unsigned int  starting_tree_level)
  {
    const auto parallel_tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&tria);

    GridTools::Cache<dim> cached_tria(tria);
    Assert(parallel_tria->n_active_cells() > 0, ExcInternalError());

    const MPI_Comm     comm = parallel_tria->get_mpi_communicator();
    ConditionalOStream pcout(std::cout,
                             (Utilities::MPI::this_mpi_process(comm) == 0));

    // Start building R-tree
    namespace bgi = boost::geometry::index;
    static constexpr unsigned int max_elem_per_node =
      constexpr_pow(2, dim); // 2^dim
    std::vector<std::pair<BoundingBox<dim>,
                          typename Triangulation<dim>::active_cell_iterator>>
                 boxes(parallel_tria->n_locally_owned_active_cells());
    unsigned int i = 0;
    for (const auto &cell : parallel_tria->active_cell_iterators())
      if (cell->is_locally_owned())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

    auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
    Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
    pcout << "Total number of available levels: " << n_levels(tree)
          << std::endl;

    pcout << "Starting level: " << starting_tree_level << std::endl;
    const unsigned int total_tree_levels =
      n_levels(tree) - starting_tree_level + 1;


    // Resize the agglomeration handlers to the right size

    agglomeration_handlers.resize(total_tree_levels);
    // Loop through the available levels and set AgglomerationHandlers up.
    for (unsigned int extraction_level = starting_tree_level;
         extraction_level <= n_levels(tree);
         ++extraction_level)
      {
        agglomeration_handlers[extraction_level - starting_tree_level] =
          std::make_unique<AgglomerationHandler<dim>>(cached_tria);
        CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                            extraction_level};
        const auto agglomerates = agglomerator.extract_agglomerates();
        agglomeration_handlers[extraction_level - starting_tree_level]
          ->connect_hierarchy(agglomerator);

        // Flag elements for agglomeration
        unsigned int agglo_index = 0;
        for (unsigned int i = 0; i < agglomerates.size(); ++i)
          {
            const auto &agglo = agglomerates[i]; // i-th agglomerate
            for (const auto &el : agglo)
              {
                el->set_material_id(agglo_index);
              }
            ++agglo_index;
          }

        const unsigned int n_local_agglomerates = agglo_index;
        unsigned int       total_agglomerates =
          Utilities::MPI::sum(n_local_agglomerates, comm);
        pcout << "Total agglomerates per (tree) level: " << extraction_level
              << ": " << total_agglomerates << std::endl;


        // Now, perform agglomeration within each locally owned partition
        std::vector<
          std::vector<typename Triangulation<dim>::active_cell_iterator>>
          cells_per_subdomain(n_local_agglomerates);
        for (const auto &cell : parallel_tria->active_cell_iterators())
          if (cell->is_locally_owned())
            cells_per_subdomain[cell->material_id()].push_back(cell);

        // For every subdomain, agglomerate elements together
        for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
          agglomeration_handlers[extraction_level - starting_tree_level]
            ->define_agglomerate(cells_per_subdomain[i]);

        agglomeration_handlers[extraction_level - starting_tree_level]
          ->initialize_fe_values(QGauss<dim>(fe_dg.degree + 1),
                                 update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points,
                                 QGauss<dim - 1>(fe_dg.degree + 1),
                                 update_JxW_values);
        agglomeration_handlers[extraction_level - starting_tree_level]
          ->distribute_agglomerated_dofs(fe_dg);
      }

    return total_tree_levels;
  }



  /**
   * Utility to compute jump terms when the interface is locally owned, i.e.
   * both elements are locally owned.
   */
  template <int dim>
  void
  assemble_local_jumps_and_averages(FullMatrix<double>      &M11,
                                    FullMatrix<double>      &M12,
                                    FullMatrix<double>      &M21,
                                    FullMatrix<double>      &M22,
                                    const FEValuesBase<dim> &fe_faces0,
                                    const FEValuesBase<dim> &fe_faces1,
                                    const double             penalty_constant,
                                    const double             h_f)
  {
    const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
    const unsigned int                 dofs_per_cell =
      M11.m(); // size of local matrices equals the #DoFs
    for (unsigned int q_index : fe_faces0.quadrature_point_indices())
      {
        const Tensor<1, dim> &normal = normals[q_index];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                M11(i, j) += (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                                fe_faces0.shape_value(j, q_index) -
                              0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                                fe_faces0.shape_value(i, q_index) +
                              (penalty_constant / h_f) *
                                fe_faces0.shape_value(i, q_index) *
                                fe_faces0.shape_value(j, q_index)) *
                             fe_faces0.JxW(q_index);
                M12(i, j) += (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                                fe_faces1.shape_value(j, q_index) -
                              0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                                fe_faces0.shape_value(i, q_index) -
                              (penalty_constant / h_f) *
                                fe_faces0.shape_value(i, q_index) *
                                fe_faces1.shape_value(j, q_index)) *
                             fe_faces1.JxW(q_index);
                M21(i, j) += (-0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                                fe_faces0.shape_value(j, q_index) +
                              0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                                fe_faces1.shape_value(i, q_index) -
                              (penalty_constant / h_f) *
                                fe_faces1.shape_value(i, q_index) *
                                fe_faces0.shape_value(j, q_index)) *
                             fe_faces1.JxW(q_index);
                M22(i, j) += (0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                                fe_faces1.shape_value(j, q_index) +
                              0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                                fe_faces1.shape_value(i, q_index) +
                              (penalty_constant / h_f) *
                                fe_faces1.shape_value(i, q_index) *
                                fe_faces1.shape_value(j, q_index)) *
                             fe_faces1.JxW(q_index);
              }
          }
      }
  }
  /**
   * Same as above, but for a ghosted neighbor.
   */
  template <int dim>
  void
  assemble_local_jumps_and_averages_ghost(
    FullMatrix<double>                             &M11,
    FullMatrix<double>                             &M12,
    FullMatrix<double>                             &M21,
    FullMatrix<double>                             &M22,
    const FEValuesBase<dim>                        &fe_faces0,
    const std::vector<std::vector<double>>         &recv_values,
    const std::vector<std::vector<Tensor<1, dim>>> &recv_gradients,
    const std::vector<double>                      &recv_jxws,
    const double                                    penalty_constant,
    const double                                    h_f)
  {
    Assert(
      (recv_values.size() > 0 && recv_gradients.size() && recv_jxws.size()),
      ExcMessage(
        "Not possible to assemble jumps and averages at a ghosted interface."));
    const unsigned int                 dofs_per_cell = M11.m();
    const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
    for (unsigned int q_index : fe_faces0.quadrature_point_indices())
      {
        const Tensor<1, dim> &normal = normals[q_index];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                M11(i, j) += (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                                fe_faces0.shape_value(j, q_index) -
                              0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                                fe_faces0.shape_value(i, q_index) +
                              (penalty_constant / h_f) *
                                fe_faces0.shape_value(i, q_index) *
                                fe_faces0.shape_value(j, q_index)) *
                             fe_faces0.JxW(q_index);
                M12(i, j) += (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                                recv_values[j][q_index] -
                              0.5 * recv_gradients[j][q_index] * normal *
                                fe_faces0.shape_value(i, q_index) -
                              (penalty_constant / h_f) *
                                fe_faces0.shape_value(i, q_index) *
                                recv_values[j][q_index]) *
                             recv_jxws[q_index];
                M21(i, j) +=
                  (-0.5 * recv_gradients[i][q_index] * normal *
                     fe_faces0.shape_value(j, q_index) +
                   0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                     recv_values[i][q_index] -
                   (penalty_constant / h_f) * recv_values[i][q_index] *
                     fe_faces0.shape_value(j, q_index)) *
                  recv_jxws[q_index];
                M22(i, j) +=
                  (0.5 * recv_gradients[i][q_index] * normal *
                     recv_values[j][q_index] +
                   0.5 * recv_gradients[j][q_index] * normal *
                     recv_values[i][q_index] +
                   (penalty_constant / h_f) * recv_values[i][q_index] *
                     recv_values[j][q_index]) *
                  recv_jxws[q_index];
              }
          }
      }
  }


  /**
   * Utility function to assemble the SIPDG Laplace matrix.
   * @note Supported matrix types are Trilinos types and native SparseMatrix
   * objects provided by deal.II.
   */
  template <int dim, typename MatrixType>
  void
  assemble_dg_matrix(MatrixType                      &system_matrix,
                     const FiniteElement<dim>        &fe_dg,
                     const AgglomerationHandler<dim> &ah)
  {
    static_assert(
      (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix> ||
       std::is_same_v<MatrixType,
                      SparseMatrix<typename MatrixType::value_type>>));

    Assert((dynamic_cast<const FE_DGQ<dim> *>(&fe_dg) ||
            dynamic_cast<const FE_DGP<dim> *>(&fe_dg) ||
            dynamic_cast<const FE_SimplexDGP<dim> *>(&fe_dg)),
           ExcMessage("FE type not supported."));

    AffineConstraints constraints;
    constraints.close();
    const double penalty_constant =
      10 * (fe_dg.degree + dim) * (fe_dg.degree + 1);
    TrilinosWrappers::SparsityPattern dsp;
    const_cast<AgglomerationHandler<dim> &>(ah)
      .create_agglomeration_sparsity_pattern(dsp);
    system_matrix.reinit(dsp);
    const unsigned int dofs_per_cell = fe_dg.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_neighbor(
      dofs_per_cell);

    for (const auto &polytope : ah.polytope_iterators())
      {
        if (polytope->is_locally_owned())
          {
            cell_matrix              = 0.;
            const auto &agglo_values = ah.reinit(polytope);
            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          agglo_values.shape_grad(i, q_index) *
                          agglo_values.shape_grad(j, q_index) *
                          agglo_values.JxW(q_index);
                      }
                  }
              }
            // get volumetric DoFs
            polytope->get_dof_indices(local_dof_indices);
            // Assemble face terms
            unsigned int n_faces = polytope->n_faces();
            const double h_f     = polytope->diameter();
            for (unsigned int f = 0; f < n_faces; ++f)
              {
                if (polytope->at_boundary(f))
                  {
                    // Get normal vectors seen from each agglomeration.
                    const auto &fe_face = ah.reinit(polytope, f);
                    const auto &normals = fe_face.get_normal_vectors();
                    for (unsigned int q_index :
                         fe_face.quadrature_point_indices())
                      {
                        const Tensor<1, dim> &normal = normals[q_index];
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                cell_matrix(i, j) +=
                                  (-fe_face.shape_value(i, q_index) *
                                     fe_face.shape_grad(j, q_index) * normal -
                                   fe_face.shape_grad(i, q_index) * normal *
                                     fe_face.shape_value(j, q_index) +
                                   (penalty_constant / h_f) *
                                     fe_face.shape_value(i, q_index) *
                                     fe_face.shape_value(j, q_index)) *
                                  fe_face.JxW(q_index);
                              }
                          }
                      }
                  }
                else
                  {
                    const auto &neigh_polytope = polytope->neighbor(f);
                    if (polytope->id() < neigh_polytope->id())
                      {
                        unsigned int nofn =
                          polytope->neighbor_of_agglomerated_neighbor(f);
                        Assert(neigh_polytope->neighbor(nofn)->id() ==
                                 polytope->id(),
                               ExcMessage("Mismatch."));
                        const auto &fe_faces = ah.reinit_interface(
                          polytope, neigh_polytope, f, nofn);
                        const auto &fe_faces0 = fe_faces.first;
                        if (neigh_polytope->is_locally_owned())
                          {
                            // use both fevalues
                            const auto &fe_faces1 = fe_faces.second;
                            M11                   = 0.;
                            M12                   = 0.;
                            M21                   = 0.;
                            M22                   = 0.;
                            assemble_local_jumps_and_averages(M11,
                                                              M12,
                                                              M21,
                                                              M22,
                                                              fe_faces0,
                                                              fe_faces1,
                                                              penalty_constant,
                                                              h_f);
                            // distribute DoFs accordingly
                            // fluxes
                            neigh_polytope->get_dof_indices(
                              local_dof_indices_neighbor);
                            constraints.distribute_local_to_global(
                              M11, local_dof_indices, system_matrix);
                            constraints.distribute_local_to_global(
                              M12,
                              local_dof_indices,
                              local_dof_indices_neighbor,
                              system_matrix);
                            constraints.distribute_local_to_global(
                              M21,
                              local_dof_indices_neighbor,
                              local_dof_indices,
                              system_matrix);
                            constraints.distribute_local_to_global(
                              M22, local_dof_indices_neighbor, system_matrix);
                          }
                        else
                          {
                            // neigh polytope is ghosted, so retrieve necessary
                            // metadata.
                            types::subdomain_id neigh_rank =
                              neigh_polytope->subdomain_id();
                            const auto &recv_jxws =
                              ah.recv_jxws.at(neigh_rank)
                                .at({neigh_polytope->id(), nofn});
                            const auto &recv_values =
                              ah.recv_values.at(neigh_rank)
                                .at({neigh_polytope->id(), nofn});
                            const auto &recv_gradients =
                              ah.recv_gradients.at(neigh_rank)
                                .at({neigh_polytope->id(), nofn});
                            M11 = 0.;
                            M12 = 0.;
                            M21 = 0.;
                            M22 = 0.;
                            // there's no FEFaceValues on the other side (it's
                            // ghosted), so we just pass the actual data we have
                            // recevied from the neighboring ghosted polytope
                            assemble_local_jumps_and_averages_ghost(
                              M11,
                              M12,
                              M21,
                              M22,
                              fe_faces0,
                              recv_values,
                              recv_gradients,
                              recv_jxws,
                              penalty_constant,
                              h_f);
                            // distribute DoFs accordingly
                            // fluxes
                            neigh_polytope->get_dof_indices(
                              local_dof_indices_neighbor);
                            constraints.distribute_local_to_global(
                              M11, local_dof_indices, system_matrix);
                            constraints.distribute_local_to_global(
                              M12,
                              local_dof_indices,
                              local_dof_indices_neighbor,
                              system_matrix);
                            constraints.distribute_local_to_global(
                              M21,
                              local_dof_indices_neighbor,
                              local_dof_indices,
                              system_matrix);
                            constraints.distribute_local_to_global(
                              M22, local_dof_indices_neighbor, system_matrix);
                          } // ghosted polytope case
                      }     // only once
                  }         // internal face
              }             // face loop
            constraints.distribute_local_to_global(cell_matrix,
                                                   local_dof_indices,
                                                   system_matrix);
          } // locally owned polytopes
      }
    system_matrix.compress(VectorOperation::add);
  }



  /**
   * Compute SIPDG matrix as well as rhs vector.
   * @note Hardcoded for $f=1$ and simplex elements.
   * TODO: Pass Function object for boundary conditions and forcing term.
   */
  template <int dim, typename MatrixType, typename VectorType>
  void
  assemble_dg_matrix_on_standard_mesh(MatrixType               &system_matrix,
                                      VectorType               &system_rhs,
                                      const Mapping<dim>       &mapping,
                                      const FiniteElement<dim> &fe_dg,
                                      const DoFHandler<dim>    &dof_handler)
  {
    static_assert(
      (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix> ||
       std::is_same_v<MatrixType,
                      SparseMatrix<typename MatrixType::value_type>>));

    Assert((dynamic_cast<const FE_SimplexDGP<dim> *>(&fe_dg) != nullptr),
           ExcNotImplemented(
             "Implemented only for simplex meshes for the time being."));

    Assert(dof_handler.get_triangulation().all_reference_cells_are_simplex(),
           ExcNotImplemented());

    const double penalty_constant = .5 * fe_dg.degree * (fe_dg.degree + 1);
    AffineConstraints<typename MatrixType::value_type> constraints;
    constraints.close();

    const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.locally_owned_dofs(),
      dof_handler.get_mpi_communicator(),
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         dof_handler.get_mpi_communicator());

    system_rhs.reinit(locally_owned_dofs, dof_handler.get_mpi_communicator());

    const unsigned int quadrature_degree = fe_dg.degree + 1;
    FEFaceValues<dim>  fe_faces0(mapping,
                                fe_dg,
                                QGaussSimplex<dim - 1>(quadrature_degree),
                                update_values | update_JxW_values |
                                  update_gradients | update_quadrature_points |
                                  update_normal_vectors);


    FEValues<dim> fe_values(mapping,
                            fe_dg,
                            QGaussSimplex<dim>(quadrature_degree),
                            update_values | update_JxW_values |
                              update_gradients | update_quadrature_points);

    FEFaceValues<dim>  fe_faces1(mapping,
                                fe_dg,
                                QGaussSimplex<dim - 1>(quadrature_degree),
                                update_values | update_JxW_values |
                                  update_gradients | update_quadrature_points |
                                  update_normal_vectors);
    const unsigned int dofs_per_cell = fe_dg.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Loop over standard deal.II cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_matrix = 0.;
            cell_rhs    = 0.;

            fe_values.reinit(cell);

            // const auto &q_points = fe_values.get_quadrature_points();
            // const unsigned int n_qpoints = q_points.size();

            for (unsigned int q_index : fe_values.quadrature_point_indices())
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                             fe_values.shape_grad(j, q_index) *
                                             fe_values.JxW(q_index);
                      }
                    cell_rhs(i) +=
                      fe_values.shape_value(i, q_index) * 1. *
                      fe_values.JxW(q_index); // TODO: pass functional
                  }
              }

            // distribute volumetric DoFs
            cell->get_dof_indices(local_dof_indices);
            double hf = 0.;
            for (const auto f : cell->face_indices())
              {
                const double extent1 =
                  cell->measure() / cell->face(f)->measure();

                if (cell->face(f)->at_boundary())
                  {
                    hf = (1. / extent1 + 1. / extent1);
                    fe_faces0.reinit(cell, f);

                    const auto &normals = fe_faces0.get_normal_vectors();
                    for (unsigned int q_index :
                         fe_faces0.quadrature_point_indices())
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                cell_matrix(i, j) +=
                                  (-fe_faces0.shape_value(i, q_index) *
                                     fe_faces0.shape_grad(j, q_index) *
                                     normals[q_index] -
                                   fe_faces0.shape_grad(i, q_index) *
                                     normals[q_index] *
                                     fe_faces0.shape_value(j, q_index) +
                                   (penalty_constant * hf) *
                                     fe_faces0.shape_value(i, q_index) *
                                     fe_faces0.shape_value(j, q_index)) *
                                  fe_faces0.JxW(q_index);
                              }
                            cell_rhs(i) +=
                              0.; // TODO: add bdary conditions functional
                          }
                      }
                  }
                else
                  {
                    const auto &neigh_cell = cell->neighbor(f);
                    if (cell->global_active_cell_index() <
                        neigh_cell->global_active_cell_index())
                      {
                        const double extent2 =
                          neigh_cell->measure() /
                          neigh_cell->face(cell->neighbor_of_neighbor(f))
                            ->measure();
                        hf = (1. / extent1 + 1. / extent2);
                        fe_faces0.reinit(cell, f);
                        fe_faces1.reinit(neigh_cell,
                                         cell->neighbor_of_neighbor(f));

                        std::vector<types::global_dof_index>
                          local_dof_indices_neighbor(dofs_per_cell);

                        M11 = 0.;
                        M12 = 0.;
                        M21 = 0.;
                        M22 = 0.;

                        const auto &normals = fe_faces0.get_normal_vectors();
                        // M11
                        for (unsigned int q_index :
                             fe_faces0.quadrature_point_indices())
                          {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                  {
                                    M11(i, j) +=
                                      (-0.5 * fe_faces0.shape_grad(i, q_index) *
                                         normals[q_index] *
                                         fe_faces0.shape_value(j, q_index) -
                                       0.5 * fe_faces0.shape_grad(j, q_index) *
                                         normals[q_index] *
                                         fe_faces0.shape_value(i, q_index) +
                                       (penalty_constant * hf) *
                                         fe_faces0.shape_value(i, q_index) *
                                         fe_faces0.shape_value(j, q_index)) *
                                      fe_faces0.JxW(q_index);

                                    M12(i, j) +=
                                      (0.5 * fe_faces0.shape_grad(i, q_index) *
                                         normals[q_index] *
                                         fe_faces1.shape_value(j, q_index) -
                                       0.5 * fe_faces1.shape_grad(j, q_index) *
                                         normals[q_index] *
                                         fe_faces0.shape_value(i, q_index) -
                                       (penalty_constant * hf) *
                                         fe_faces0.shape_value(i, q_index) *
                                         fe_faces1.shape_value(j, q_index)) *
                                      fe_faces1.JxW(q_index);

                                    // A10
                                    M21(i, j) +=
                                      (-0.5 * fe_faces1.shape_grad(i, q_index) *
                                         normals[q_index] *
                                         fe_faces0.shape_value(j, q_index) +
                                       0.5 * fe_faces0.shape_grad(j, q_index) *
                                         normals[q_index] *
                                         fe_faces1.shape_value(i, q_index) -
                                       (penalty_constant * hf) *
                                         fe_faces1.shape_value(i, q_index) *
                                         fe_faces0.shape_value(j, q_index)) *
                                      fe_faces1.JxW(q_index);

                                    // A11
                                    M22(i, j) +=
                                      (0.5 * fe_faces1.shape_grad(i, q_index) *
                                         normals[q_index] *
                                         fe_faces1.shape_value(j, q_index) +
                                       0.5 * fe_faces1.shape_grad(j, q_index) *
                                         normals[q_index] *
                                         fe_faces1.shape_value(i, q_index) +
                                       (penalty_constant * hf) *
                                         fe_faces1.shape_value(i, q_index) *
                                         fe_faces1.shape_value(j, q_index)) *
                                      fe_faces1.JxW(q_index);
                                  }
                              }
                          }

                        // distribute DoFs accordingly

                        neigh_cell->get_dof_indices(local_dof_indices_neighbor);

                        constraints.distribute_local_to_global(
                          M11, local_dof_indices, system_matrix);
                        constraints.distribute_local_to_global(
                          M12,
                          local_dof_indices,
                          local_dof_indices_neighbor,
                          system_matrix);
                        constraints.distribute_local_to_global(
                          M21,
                          local_dof_indices_neighbor,
                          local_dof_indices,
                          system_matrix);
                        constraints.distribute_local_to_global(
                          M22, local_dof_indices_neighbor, system_matrix);

                      } // check idx neighbors
                  }     // over faces
              }
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


} // namespace dealii::PolyUtils

#endif
