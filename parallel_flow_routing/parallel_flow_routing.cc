/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Colorado State University, 2026.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>



// This namespace contains the implementation of the parallel flow routing
// program. The program solves a model of water flow on terrain using the
// "downhill flow" method commonly used in hydrology. In this approach, water
// flows from each grid point in the steepest downhill direction determined by
// the terrain elevation, and the water flux distribution is computed by solving
// a steady-state flow conservation system in parallel.
namespace ParallelFlowRouting
{
  using namespace dealii;

  // The LA namespace encapsulates the linear algebra library configuration.
  // We use either PETSc or Trilinos for distributed sparse matrices and
  // vectors, depending on what deal.II was compiled with. PETSc is preferred
  // if available (and not using complex numbers), otherwise we fall back to
  // Trilinos. This choice allows for efficient parallel solving of the large
  // sparse systems that arise from discretizing the flow conservation
  // equations.
  namespace LA
  {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
    using namespace LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
  } // namespace LA

  // We use block vectors and block sparse matrices to represent the distributed
  // linear systems. Block structures allow us to organize data logically and
  // can improve performance in certain scenarios.
  using VectorType = LA::MPI::BlockVector;
  using MatrixType = LA::MPI::BlockSparseMatrix;


  // @sect3{ColoradoTopography}
  //
  // This class represents the topographic elevation data for Colorado, defined
  // on the domain spanning 7 degrees longitude by 4 degrees latitude
  // (from 109°W to 102°W and from 37°N to 41°N). The class reads elevation data
  // from a gzip-compressed file, processes it to ensure it is suitable for flow
  // routing (by filling depressions), and then caches the processed data for
  // reuse in subsequent runs.
  //
  // The elevation data is provided at a baseline resolution of 1800 meters per
  // pixel. The n_refinements parameter allows scaling to finer resolutions by
  // subdividing each baseline grid cell into 2^n_refinements smaller cells.
  // This is useful for studying the flow routing algorithm at different
  // resolutions.
  //
  // The class inherits from the deal.II Function class, so it can be used
  // directly with the VectorTools::interpolate() function to set elevation
  // values on a finite element mesh. The function is evaluated on 3D mesh
  // points by first converting Cartesian coordinates (used on the mesh) to
  // geographic coordinates (longitude and latitude in degrees), and then
  // interpolating the stored elevation data.
  //
  // A key feature of this class is the removal of local depressions (sinks)
  // from the elevation data. These depressions are problematic for flow routing
  // because water trapped in them does not flow anywhere. The class uses a
  // priority-flood depression-filling algorithm to ensure that the resulting
  // digital elevation model (DEM) has no interior depressions. This is done
  // during construction in parallel, with process 0 performing the computation
  // and broadcasting the result to all other processes.
  class ColoradoTopography : public Function<3>
  {
  public:
    // Constructor that loads and processes the Colorado topography data.
    // - `mpi_communicator`: The MPI communicator used for parallel
    //   communication. Process 0 reads and processes the data, then
    //   broadcasts it to all others.
    // - `n_refinements`: The refinement level. The mesh will have
    //   7*2^n_refinements by 4*2^n_refinements cells.
    ColoradoTopography(const MPI_Comm     mpi_communicator,
                       const unsigned int n_refinements);

    // Return the elevation in meters at a given 3D point on the Earth's
    // surface. The point is first converted from Cartesian coordinates (as used
    // on the mesh) to geographic coordinates (longitude and latitude in
    // degrees), and then the stored elevation data is interpolated.
    virtual double
    value(const Point<3> &p,
          const unsigned int /*component*/ = 0) const override;

  private:
    // The InterpolatedUniformGridData object stores the actual elevation values
    // on a uniform grid in longitude-latitude space.
    std::unique_ptr<const Functions::InterpolatedUniformGridData<2>> data;

    // Count the number of local depressions in an elevation table. A depression
    // is a grid point that is lower than all of its 8 neighbors (including
    // diagonals).
    static unsigned int
    count_depressions(const Table<2, double> &elevation_data);

    // Fill (eliminate) local depressions in an elevation table using a
    // priority-flood algorithm. After this function completes, no interior grid
    // point will be lower than all of its 8 neighbors.
    static void
    fill_depressions(Table<2, double> &elevation_data);
  };


  // @sect3{ColoradoTopography::ColoradoTopography()}
  //
  // The constructor is responsible for loading the Colorado topography data.
  // The procedure it follows is:
  //
  // 1. On process 0 (the root MPI process), check if a cache file exists for
  //    the requested resolution. If not, read the original data from a
  //    gzip-compressed file, interpolate it to the desired resolution, fill
  //    depressions, and cache the result in a binary-serialized file. If the
  //    cache exists, read directly from it (this is much faster).
  //
  // 2. Broadcast the elevation data to all MPI processes.
  //
  // 3. Create an InterpolatedUniformGridData object that can efficiently
  //    evaluate elevation at arbitrary points via bilinear interpolation.
  //
  // The input data file is expected to be in ESRI ASCII raster format,
  // compressed with gzip. The format includes header lines specifying the
  // number of columns and rows, the corner coordinates, and the cell size.
  ColoradoTopography::ColoradoTopography(const MPI_Comm     mpi_communicator,
                                         const unsigned int n_refinements)
  {
    unsigned int n_latitudes       = numbers::invalid_unsigned_int;
    unsigned int n_longitudes      = numbers::invalid_unsigned_int;
    Point<2>     lower_left_corner = numbers::signaling_nan<Point<2>>();
    double       pixel_size        = numbers::signaling_nan<double>();

    Table<2, double> elevation_data;

    const unsigned int root_process = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == root_process)
      {
        const std::string cache_filename =
          "colorado-topography-1800m.cache" + std::to_string(n_refinements);

        if (!std::filesystem::exists(cache_filename))
          {
            const std::string original_data_filename =
              "colorado-topography-1800m.txt.gz";
            std::cout << "   Reading original elevation data from file "
                      << original_data_filename << std::endl;

            unsigned int n_original_latitudes;
            unsigned int n_original_longitudes;
            Point<2>     original_lower_left_corner;
            double       original_pixel_size;

            Table<2, double> original_elevation_data;

            boost::iostreams::filtering_istream in;
            in.push(boost::iostreams::basic_gzip_decompressor<>());
            in.push(boost::iostreams::file_source(original_data_filename));

            std::string word;

            in >> word;
            AssertThrow(word == "ncols",
                        ExcMessage(
                          "The first line of the input file needs to start "
                          "with the word 'ncols', but starts with '" +
                          word + "'."));
            in >> n_original_longitudes;

            in >> word;
            AssertThrow(word == "nrows",
                        ExcMessage(
                          "The second line of the input file needs to start "
                          "with the word 'nrows', but starts with '" +
                          word + "'."));
            in >> n_original_latitudes;

            in >> word;
            AssertThrow(word == "xllcorner",
                        ExcMessage(
                          "The third line of the input file needs to start "
                          "with the word 'xllcorner', but starts with '" +
                          word + "'."));
            in >> original_lower_left_corner[0];

            in >> word;
            AssertThrow(word == "yllcorner",
                        ExcMessage(
                          "The fourth line of the input file needs to start "
                          "with the word 'yllcorner', but starts with '" +
                          word + "'."));
            in >> original_lower_left_corner[1];


            in >> word;
            AssertThrow(word == "cellsize",
                        ExcMessage(
                          "The fourth line of the input file needs to start "
                          "with the word 'cellsize', but starts with '" +
                          word + "'."));
            in >> original_pixel_size;

            original_elevation_data.reinit(n_original_longitudes,
                                           n_original_latitudes);

            // Data is provided in the input file as horizontal strips, with
            // longitude marching fastest from west to east. The second
            // coordinate is latitude, but the file marches north to south,
            // which is the opposite of how we want it (we want a right-handed
            // coordinate system), so we have to revert the order in which we
            // insert things into the data table.
            for (unsigned int latitude_index = 0;
                 latitude_index < n_original_latitudes;
                 ++latitude_index)
              for (unsigned int longitude_index = 0;
                   longitude_index < n_original_longitudes;
                   ++longitude_index)
                {
                  try
                    {
                      double elevation;
                      in >> elevation;

                      original_elevation_data(longitude_index,
                                              n_original_latitudes -
                                                latitude_index - 1) = elevation;
                    }
                  catch (...)
                    {
                      AssertThrow(false,
                                  ExcMessage(
                                    "Could not read all expected data points "
                                    "from the file <" +
                                    original_data_filename + ">!"));
                    }
                }

            const Functions::InterpolatedUniformGridData<2>
              original_elevation_field(
                std::array<std::pair<double, double>, 2>{
                  {std::make_pair(original_lower_left_corner[0],
                                  original_lower_left_corner[0] +
                                    (n_original_longitudes - 1) *
                                      original_pixel_size),
                   std::make_pair(original_lower_left_corner[1],
                                  original_lower_left_corner[1] +
                                    (n_original_latitudes - 1) *
                                      original_pixel_size)}},
                std::array<unsigned int, 2>{
                  {n_original_longitudes - 1, n_original_latitudes - 1}},
                std::move(original_elevation_data));

            // The model that we just read is provided on a domain that is
            // slightly larger than what we actually need, and on a mesh that
            // does not align with the vertices we will create later on. It may
            // also contain local depressions that prevent us from performing
            // useful flow routing, and even if it doesn't, the interpolation
            // onto a concrete mesh that has different vertices will create a
            // model with local depressions.
            //
            // To avoid this, we take the following steps where 'r' is a
            // parameter that controls the resolution of the mesh we will
            // create:
            // * We interpolate things onto a mesh that has 7*2^r x 4*2^r cells
            //   (i.e., 7*2^r+1 x 4*2^r+1 points) and that has the exact right
            //   extents. This makes sense because below we will
            //   start all computations on a 7x4 mesh, given that Colorado spans
            //   7 by 4 degrees on the surface of the Earth.
            // * We then perform depression filling by lifting points that are
            //   lower than all of their neighbors above the lowest of its
            //   neighbors. The result is a digital elevation model without
            //   local depressions.
            const unsigned int n_subdivisions = (1 << n_refinements);
            n_longitudes                      = 7 * n_subdivisions + 1;
            n_latitudes                       = 4 * n_subdivisions + 1;
            pixel_size                        = 1. / n_subdivisions;
            lower_left_corner = {-109., 37.}; // 109 degrees W, 37 degrees N

            elevation_data.reinit(n_longitudes, n_latitudes);
            for (unsigned int latitude_index = 0; latitude_index < n_latitudes;
                 ++latitude_index)
              for (unsigned int longitude_index = 0;
                   longitude_index < n_longitudes;
                   ++longitude_index)
                {
                  const double longitude =
                    lower_left_corner[0] + longitude_index * pixel_size;
                  const double latitude =
                    lower_left_corner[1] + latitude_index * pixel_size;
                  elevation_data(longitude_index, latitude_index) =
                    original_elevation_field.value(
                      Point<2>(longitude, latitude));
                }

            // Now we need to fix up depressions in the *interior* of the table.
            std::cout << "   Filling in the "
                      << count_depressions(elevation_data)
                      << " depressions in the elevation model" << std::endl;
            fill_depressions(elevation_data);

            // Check that we have no depressions left:
            Assert(count_depressions(elevation_data) == 0, ExcInternalError());

            // Write the elevation data to the cache file using binary
            // serialization
            std::cout << "   Writing " << elevation_data.size()[0] << " x "
                      << elevation_data.size()[1]
                      << " elevation points to cache file " << cache_filename
                      << std::endl;
            boost::iostreams::filtering_ostream out;
            out.push(boost::iostreams::basic_gzip_compressor<>());
            out.push(boost::iostreams::file_sink(cache_filename));
            boost::archive::binary_oarchive oa(out);
            oa << n_longitudes << n_latitudes << lower_left_corner << pixel_size
               << elevation_data;
          }
        else
          {
            // If we did find a cache file read by a previous run of the
            // program,
            // read the elevation data using binary deserialization
            std::cout << "   Reading elevation data from cache file "
                      << cache_filename << std::endl;

            boost::iostreams::filtering_istream in;
            in.push(boost::iostreams::basic_gzip_decompressor<>());
            in.push(boost::iostreams::file_source(cache_filename));
            boost::archive::binary_iarchive ia(in);
            ia >> n_longitudes >> n_latitudes >> lower_left_corner >>
              pixel_size >> elevation_data;

            std::cout << "   Read " << elevation_data.size()[0] << " x "
                      << elevation_data.size()[1]
                      << " elevation points from cache file" << std::endl;
          }
      }

    // Finally, distribute the data created on process 0 to everyone else and
    // create a function object that can be used to evaluate the elevation at
    // arbitrary points.
    n_latitudes =
      Utilities::MPI::broadcast(mpi_communicator, n_latitudes, root_process);
    n_longitudes =
      Utilities::MPI::broadcast(mpi_communicator, n_longitudes, root_process);
    lower_left_corner = Utilities::MPI::broadcast(mpi_communicator,
                                                  lower_left_corner,
                                                  root_process);
    pixel_size =
      Utilities::MPI::broadcast(mpi_communicator, pixel_size, root_process);

    elevation_data.replicate_across_communicator(mpi_communicator,
                                                 root_process);

    data = std::make_unique<const Functions::InterpolatedUniformGridData<2>>(
      std::array<std::pair<double, double>, 2>{
        {std::make_pair(lower_left_corner[0],
                        lower_left_corner[0] + (n_longitudes - 1) * pixel_size),
         std::make_pair(lower_left_corner[1],
                        lower_left_corner[1] +
                          (n_latitudes - 1) * pixel_size)}},
      std::array<unsigned int, 2>{{n_longitudes - 1, n_latitudes - 1}},
      std::move(elevation_data));
  }


  // @sect3{ColoradoTopography::value()}
  //
  // This function evaluates the elevation at a given 3D point. Since the mesh
  // is embedded in 3D on the surface of the Earth (as a sphere of radius 6371
  // km), the input point p is given in Cartesian coordinates. We convert these
  // to geographic coordinates (longitude and latitude in degrees) and then look
  // up the elevation in the stored data table using bilinear interpolation.
  //
  // The conversion from Cartesian to geographic coordinates uses standard
  // formulas:
  // - Longitude (x-y plane angle): atan2(y, x) * 360 / (2π)
  // - Latitude (z-radial angle): atan2(z, sqrt(x² + y²)) * 360 / (2π)
  double
  ColoradoTopography::value(const Point<3> &p,
                            const unsigned int /*component*/) const
  {
    // First pull back p to longitude/latitude, expressed in degrees
    const Point<2> p_long_lat(std::atan2(p[1], p[0]) * 360 / (2 * numbers::PI),

                              std::atan2(p[2],
                                         std::sqrt(p[0] * p[0] + p[1] * p[1])) *
                                360 / (2 * numbers::PI));

    return data->value(p_long_lat);
  }


  // @sect3{ColoradoTopography::count_depressions()}
  //
  // This static helper function counts the number of local depressions in an
  // elevation table. A local depression is a grid point in the interior of the
  // domain that is lower than all of its 8 neighbors (i.e., all
  // immediate neighbors including diagonals).
  //
  // We only check interior points (excluding the boundary of the domain)
  // because boundary points can have lower neighbors (water can flow out of the
  // domain at the boundary at these points).
  unsigned int
  ColoradoTopography::count_depressions(const Table<2, double> &elevation_data)
  {
    const unsigned int n_longitudes = elevation_data.size()[0];
    const unsigned int n_latitudes  = elevation_data.size()[1];

    unsigned int n_depressions = 0;
    for (unsigned int x = 1; x < n_longitudes - 1; ++x)
      for (unsigned int y = 1; y < n_latitudes - 1; ++y)
        {
          const double elevation        = elevation_data(x, y);
          double min_neighbor_elevation = std::numeric_limits<double>::max();
          for (int i = -1; i <= +1; ++i)
            for (int j = -1; j <= +1; ++j)
              if (!(i == 0 && j == 0))
                min_neighbor_elevation = std::min(min_neighbor_elevation,
                                                  elevation_data(x + i, y + j));
          if (min_neighbor_elevation >= elevation)
            ++n_depressions;
        }
    return n_depressions;
  }


  // @sect3{ColoradoTopography::fill_depressions()}
  //
  // This function removes local depressions (sinks) from a gridded elevation
  // model using a priority-flood algorithm. In topographic data, depressions
  // are local minima that are lower than all their neighbors; they are
  // problematic for flow routing because they trap water that would
  // otherwise flow downhill.
  //
  // The key idea is to work "inward" from the boundaries: we start by marking
  // all border cells as processed and placing them in a priority queue sorted
  // by elevation (lowest first). Then, we repeatedly extract the lowest cell
  // from the queue and examine its unprocessed 8-connected neighbors
  // (including diagonals). For each unprocessed neighbor, we compute its
  // filled elevation as the maximum of its current elevation and the parent
  // cell's elevation. To avoid creating perfectly flat plateaus that would
  // be ambiguous for flow routing, we add a small deterministic increment
  // (derived from the cell's grid indices). We then mark the neighbor as
  // processed, update its elevation in-place, and add it to the queue. This
  // continues until the queue is empty.
  //
  // The result is that no interior cell is lower than all of its neighbors.
  // The algorithm runs in O(N log N) time where N is the number of grid cells,
  // and the deterministic increment ensures reproducibility across runs.
  //
  // @note In hindsight, the choice of a random increment between zero and
  //  0.1 may have been a bit large -- we really just want to avoid flat
  //  plateaus, so an increment on the order of 0.01 or even smaller would have
  //  been sufficient. The current choice may create some small artificial
  //  slopes that might add up to too much elevation change across the domain.
  //  But, this is the value used for the experiments in the accompanying paper,
  //  so we keep it as is for now.
  void
  ColoradoTopography::fill_depressions(Table<2, double> &elevation_data)
  {
    struct Node
    {
      unsigned int x, y;
      double       elev;
      bool
      operator>(const Node &other) const
      {
        return elev > other.elev;
      }
    };

    const unsigned int n_rows = elevation_data.size()[0];
    const unsigned int n_cols = elevation_data.size()[1];
    Table<2, bool>     processed(n_rows, n_cols);
    processed.fill(false);

    // A priority queue that always gives us the lowest elevation node that
    // has not yet been processed. std::priority_queue is a rarely used
    // data structure that is a heap-based implementation of a priority queue.
    // You can find its description at
    // https://en.cppreference.com/w/cpp/container/priority_queue.html
    std::priority_queue<Node,
                        std::vector<Node>,
                        /* sort low-to-high */ std::greater<Node>>
      currently_active_nodes;

    std::mt19937 rng;

    // Push all border nodes into the priority queue
    for (unsigned int i = 0; i < n_rows; ++i)
      {
        currently_active_nodes.push(Node{i, 0, elevation_data[i][0]});
        currently_active_nodes.push(
          Node{i, n_cols - 1, elevation_data[i][n_cols - 1]});
        processed[i][0] = processed[i][n_cols - 1] = true;
      }
    for (unsigned int j = 0; j < n_cols; ++j)
      {
        currently_active_nodes.push(Node{0, j, elevation_data[0][j]});
        currently_active_nodes.push(
          Node{n_rows - 1, j, elevation_data[n_rows - 1][j]});
        processed[0][j] = processed[n_rows - 1][j] = true;
      }

    // Directions for 8 neighbors
    const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    // While there are nodes for which we can still look for neighbors:
    while (!currently_active_nodes.empty())
      {
        const Node current_node =
          currently_active_nodes
            .top(); // get the lowest point currently in the queue
        currently_active_nodes.pop(); // and then remove it from the queue

        // Loop over the neighbors of the lowest point in the queue, excluding
        // points that are beyond the boundary of the domain and also skipping
        // over the ones that have already been processed:
        for (unsigned int k = 0; k < 8; ++k)
          if ((static_cast<signed int>(current_node.x) + dx[k] >= 0) &&
              (static_cast<signed int>(current_node.x) + dx[k] <
               static_cast<signed int>(n_rows)) &&
              (static_cast<signed int>(current_node.y) + dy[k] >= 0) &&
              (static_cast<signed int>(current_node.y) + dy[k] <
               static_cast<signed int>(n_cols)))

            {
              const unsigned int neighbor_x = current_node.x + dx[k];
              const unsigned int neighbor_y = current_node.y + dy[k];

              if (processed[neighbor_x][neighbor_y])
                continue;

              // If the neighbor exists and has not been processed:
              // Add a random increment between 0 and 0.1 meters to ensure
              // that we do not create perfectly flat plateaus. The increment is
              // random but deterministic to ensure reproducibility across runs.
              const double new_elevation =
                std::max(elevation_data[neighbor_x][neighbor_y],
                         current_node.elev +
                           std::uniform_real_distribution<>(0, 0.1)(rng));
              elevation_data[neighbor_x][neighbor_y] = new_elevation;
              processed[neighbor_x][neighbor_y]      = true;

              // Push that neighbor to the queue:
              currently_active_nodes.push(
                {static_cast<unsigned int>(neighbor_x),
                 static_cast<unsigned int>(neighbor_y),
                 new_elevation});
            }
      }
  }



  // @sect3{RainFallRate}
  //
  // This is a simple class that describes the rain fall rate on the domain.
  // In reality, the rain fall rate varies depending on location and climate
  // conditions, but for this program we use a constant value everywhere on
  // the domain. The rain fall rate is an important boundary condition for
  // the flow routing problem: it represents the water that enters the system
  // through precipitation, and eventually either flows out through the
  // boundary or accumulates in local depressions.
  //
  // The value of 375 mm per year (approximately 15 inches per year) is
  // roughly representative of the rainfall in Colorado.
  template <int spacedim>
  class RainFallRate : public Function<spacedim>
  {
  public:
    virtual double
    value(const Point<spacedim> &p,
          const unsigned int     component = 0) const override;
  };


  template <int spacedim>
  double
  RainFallRate<spacedim>::value(const Point<spacedim> & /*p*/,
                                const unsigned int /*component*/) const
  {
    return 0.375;
  }



  /**
   * This is the primary class of this file. It implements a distributed-memory
   * parallel solver for water flow
   * routing on large-scale topographic domains. The key idea is to simulate how
   * water accumulates and flows across an elevation field, a fundamental
   * problem in terrain analysis and hydrology.
   *
   * The ParallelFlowRouter works as follows:
   *
   * 1. **Setup**: The code loads a high-resolution Colorado topography dataset
   *    and distributes it across multiple MPI processes using domain
   *    decomposition. The elevation field is discretized using bilinear finite
   *    elements.
   *
   * 2. **Elevation Processing**: The topography is preprocessed to fill local
   *    depressions (sinks) that would otherwise block water flow. This step is
   *    crucial for realistic flow routing on real terrain.
   *
   * 3. **Flow Direction Computation**: Based on the elevation gradients, the
   *    code computes the steepest descent direction (downhill direction) at
   *    each node. This creates a local flow routing pattern where water moves
   *    from higher to lower elevations.
   *
   * 4. **Flow Accumulation**: The solver assembles a large system matrix that
   *    represents how water flows from one element to its neighbors. Water that
   *    enters an element from upstream must equal the water leaving to
   *    downstream neighbors. This conservation principle is enforced via a
   *    linear system.
   *
   * 5. **Solver Strategy**: Due to the system's irregular structure
   *    (matrix-free operators and special block structure), the code uses a
   *    custom preconditioner combined with Richardson iteration to solve the
   *    flow routing equations in parallel, using the four methods outlined
   *    in the accompanying paper.
   *
   * 6. **Output**: Finally, the accumulated water flow is visualized in VTU
   *    format, showing how water accumulates in valleys and river networks.
   */
  class ParallelFlowRouter : public ParameterAcceptor
  {
  public:
    static constexpr int dim      = 2;
    static constexpr int spacedim = 3;

    ParallelFlowRouter();

    void
    run();

  private:
    void
    make_grid();

    void
    setup_dofs();

    void
    interpolate_initial_elevation();

    void
    sort_dofs_high_to_low();

    void
    compute_local_flow_routing();

    void
    assemble_system();

    void
    assemble_matrix_free_operators();

    void
    solve();

    void
    check_conservation_for_waterflow_system(const VectorType &solution);

    void
    output_results();

    static constexpr FEValuesExtractors::Scalar elevation =
      FEValuesExtractors::Scalar(0);
    static constexpr FEValuesExtractors::Scalar water_flow_rate =
      FEValuesExtractors::Scalar(1);

    const MPI_Comm mpi_communicator;

    unsigned int n_refinements;
    bool         generate_graphical_output;

    const FESystem<dim, spacedim>                       fe;
    parallel::distributed::Triangulation<dim, spacedim> triangulation;
    DoFHandler<dim, spacedim>                           dof_handler;

    IndexSet              locally_relevant_dofs;
    std::vector<IndexSet> locally_owned_partitioning;
    std::vector<IndexSet> locally_relevant_partitioning;

    IndexSet locally_relevant_water_dofs;

    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      local_flow_routing;

    MatrixType system_matrix;
    VectorType locally_relevant_solution;
    VectorType locally_relevant_solution_dot;
    VectorType system_rhs;

    class FlowRoutingMatrix;
    std::unique_ptr<const FlowRoutingMatrix> flow_routing_matrix;

    class FlowRoutingPreconditioner;
    std::unique_ptr<const FlowRoutingPreconditioner>
      flow_routing_preconditioner;

    class IplusminusXMatrixBase;

    class IplusXMatrix;
    std::unique_ptr<const IplusXMatrix> I_plus_X_matrix;

    class IminusXMatrix;
    std::unique_ptr<const IminusXMatrix> I_minus_X_matrix;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  ParallelFlowRouter::ParallelFlowRouter()
    : ParameterAcceptor("ParallelFlowRouter")
    , mpi_communicator(MPI_COMM_WORLD)
    , n_refinements(9)
    , fe(FE_Q<dim, spacedim>(1) ^ 2)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim, spacedim>::MeshSmoothing(
                      Triangulation<dim, spacedim>::smoothing_on_refinement |
                      Triangulation<dim, spacedim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {
    add_parameter("Number of refinements",
                  n_refinements,
                  "The number of global refinements to perform.");
    add_parameter("Generate graphical output",
                  generate_graphical_output,
                  "Whether to generate graphical output files.");
  }


  // @sect3{ParallelFlowRouter::make_grid()}
  //
  // This function creates the mesh that discretizes the Colorado topography
  // domain. Rather than using a flat Cartesian coordinate system, the mesh is
  // mapped onto the surface of the Earth (a sphere of radius 6371 km), so that
  // distances and areas computed on the mesh are realistic.
  //
  // The function starts by creating a rectangular mesh that spans Colorado's
  // geographic extent (7 degrees longitude × 4 degrees latitude). After
  // global refinements, this mesh is then transformed from longitude-latitude
  // coordinates to 3D Cartesian coordinates on the Earth's surface via a
  // cylindrical projection. This ensures that when we compute gradients or
  // areas on the mesh, they respect the curvature of the Earth, which is
  // essential for correct flow routing on real terrain.
  void
  ParallelFlowRouter::make_grid()
  {
    TimerOutput::Scope t(computing_timer, "Make grid");
    pcout << "Making grid... " << std::endl;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {7,
                                               4}, // Colorado spans 7x4 degrees
                                              Point<2>(-109., 37.),
                                              Point<2>(-102., 41.));

    triangulation.refine_global(n_refinements);

    GridTools::transform(
      [](const Point<spacedim> &p_long_lat_degrees) {
        const Point<2> p_long_lat(p_long_lat_degrees[0] / 360 *
                                    (2 * numbers::PI),
                                  p_long_lat_degrees[1] / 360 *
                                    (2 * numbers::PI));
        const double   R = 6371000;
        return Point<spacedim>(R * std::cos(p_long_lat[1]) *
                                 std::cos(p_long_lat[0]), // X
                               R * std::cos(p_long_lat[1]) *
                                 std::sin(p_long_lat[0]),    // Y
                               R * std::sin(p_long_lat[1])); // Z
      },
      triangulation);

    pcout << "   Number of cells: " << triangulation.n_global_active_cells()
          << std::endl;

    double area = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
      area += cell->measure();
    pcout << "   Area of the domain: " << area << "m^2" << std::endl;
  }


  // @sect3{ParallelFlowRouter::setup_system()}
  //
  // The next function also is not something that is new in any particular
  // way. Conceptually, all we have to do is set up block vectors and matrices
  // for the linear systems we want to solve. This really is quite
  // straightforward with the only complication that we have to account for
  // the fact that we are working in a parallel program where we have to
  // keep track which process owns which degrees of freedom.
  //
  // This function does the basic set-up. In following functions, we will
  // re-enumerate the degrees of freedom in a way that makes the flow routing
  // matrix have a nice triangular structure, and we will also set up the
  // matrix-free operators that we will use to compare the matrix-based and
  // matrix-free solvers.
  void
  ParallelFlowRouter::setup_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Setup system");
    pcout << "Setting up system... " << std::endl;

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

    const types::global_dof_index n_elevation_dofs      = dofs_per_block[0];
    const types::global_dof_index n_waterflow_rate_dofs = dofs_per_block[1];

    {
      const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
      locally_owned_partitioning         = {
        locally_owned_dofs.get_view(0, n_elevation_dofs),
        locally_owned_dofs.get_view(n_elevation_dofs,
                                    n_elevation_dofs + n_waterflow_rate_dofs)};
    }


    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    locally_relevant_partitioning = {
      locally_relevant_dofs.get_view(0, n_elevation_dofs),
      locally_relevant_dofs.get_view(n_elevation_dofs,
                                     n_elevation_dofs + n_waterflow_rate_dofs)};

    IndexSet all_elevation_dofs(dof_handler.n_dofs());
    all_elevation_dofs.add_range(0, n_elevation_dofs);
    locally_relevant_water_dofs = locally_relevant_dofs;
    locally_relevant_water_dofs.subtract_set(all_elevation_dofs);

    locally_relevant_solution.reinit(locally_owned_partitioning,
                                     locally_relevant_partitioning,
                                     mpi_communicator);
    locally_relevant_solution_dot.reinit(locally_owned_partitioning,
                                         locally_relevant_partitioning,
                                         mpi_communicator);
    system_rhs.reinit(locally_owned_partitioning, mpi_communicator);

    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << " (elevation: " << n_elevation_dofs
          << ", waterflow: " << n_waterflow_rate_dofs << ')' << std::endl;
    const std::vector<types::global_dof_index> water_dofs_per_process =
      Utilities::MPI::all_gather(mpi_communicator,
                                 locally_owned_partitioning[1].n_elements());
    pcout << "   Number of waterflow degrees of freedom per process: "
          << std::accumulate(water_dofs_per_process.begin(),
                             water_dofs_per_process.end(),
                             0) /
               Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " (average) x "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " (number of processes)" << std::endl;
  }


  // @sect3{ParallelFlowRouter::interpolate_initial_elevation()}
  //
  // The following function then interpolates the initial elevation onto the
  // mesh. We need this as initial conditions for the elevation variable.
  //
  // The way this function works is that given a scalar function object (derived
  // from the Function class), we first create a function object that covers
  // all solution variables, returns the elevation in one vector component
  // (specifically, in vector component zero) and zeros in all others. This
  // "extension" of a scalar to a vector function is done by the
  // VectorFunctionFromScalarFunctionObject class. We can then use this
  // extended function object to interpolate these initial conditions onto
  // all degrees of freedom, which correctly sets the initial elevation
  // variables to their initial values and sets the water flow rate to
  // zero.
  void
  ParallelFlowRouter::interpolate_initial_elevation()
  {
    TimerOutput::Scope t(computing_timer,
                         "Initial conditions: interpolate elevation");
    pcout << "Interpolating elevation... " << std::endl;

    const ColoradoTopography colorado_topography(mpi_communicator,
                                                 n_refinements);
    const VectorFunctionFromScalarFunctionObject<spacedim> initial_values(
      [&](const Point<spacedim> &p) { return colorado_topography.value(p); },
      /* elevation vector component = */ 0,
      /* total number of vector components = */ 2);

    VectorType interpolated_initial_condition(locally_owned_partitioning,
                                              MPI_COMM_WORLD);
    VectorTools::interpolate(dof_handler,
                             initial_values,
                             interpolated_initial_condition);

    // The vector we have just interpolated into is a "fully distributed
    // vector", i.e., every element is uniquely owned by one of the MPI
    // processes and these are the only ones we store on the current process.
    // On the other hand, we will also have to access values for nodes that
    // are owned by other processes (for example on ghost cells), so we copy
    // the vector into one that also has these ghost entries:
    locally_relevant_solution = interpolated_initial_condition;
  }



  // @sect3{ParallelFlowRouter::sort_dofs_high_to_low()}
  //
  // The key insight for efficiently solving the flow routing problem is that
  // water always flows downhill. This means we can process water flow
  // calculations in order from the highest elevation to the lowest. This
  // ordering is crucial for the performance of the solver: it means that when
  // we compute the water flow at a given node, all of the upstream nodes from
  // which it receives water have already been processed. This is the
  // characteristic of a triangular system, and it allows us to solve the
  // system very efficiently using a simple substitution method.
  //
  // This function implements this idea by renumbering the degrees of freedom
  // that represent the water flow rate such that they are ordered from highest
  // elevation to lowest. This doesn't change the mathematical problem we're
  // solving, but it transforms the matrix of the linear system into a lower
  // triangular matrix when water flows are processed in this order, making
  // the solver much more efficient.
  //
  // The algorithm works by first collecting all water degrees of freedom and
  // their corresponding elevations, sorting them from highest to lowest, and
  // then renumbering the DoFs accordingly. After renumbering, we update the
  // index sets used for parallel communication to reflect the new ordering.
  //
  // As the paper notes, this reordering step is not actually necessary if
  // you implement the per-process high-to-low flow routing algorithm in a
  // matrix-free way, but it is convenient for the matrix-based solver because
  // in that case, the preconditioner is a triangular solve which is exactly
  // what the Gauss-Seidel (SOR) method will perform and so we can use what
  // PETSc offers to us without having to implement a custom preconditioner.
  void
  ParallelFlowRouter::sort_dofs_high_to_low()
  {
    TimerOutput::Scope t(computing_timer,
                         "Initial conditions: sort DoFs high to low");
    pcout << "Sorting DoFs high to low... " << std::endl;

    std::map<types::global_dof_index, double> water_dof_index_to_elevation_map;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        for (unsigned int v = 0; v < cell->reference_cell().n_vertices(); ++v)
          {
            const types::global_dof_index vertex_water_dof =
              cell->vertex_dof_index(v, 1);

            if (dof_handler.locally_owned_dofs().is_element(vertex_water_dof))
              {
                const types::global_dof_index vertex_elevation_dof =
                  cell->vertex_dof_index(v, 0);
                const double vertex_elevation =
                  locally_relevant_solution(vertex_elevation_dof);
                water_dof_index_to_elevation_map[vertex_water_dof] =
                  vertex_elevation;
              }
          }

    std::vector<std::pair<types::global_dof_index, double>>
      water_dof_index_to_elevation_list(
        water_dof_index_to_elevation_map.size());
    std::copy(water_dof_index_to_elevation_map.begin(),
              water_dof_index_to_elevation_map.end(),
              water_dof_index_to_elevation_list.begin());
    std::sort(water_dof_index_to_elevation_list.begin(),
              water_dof_index_to_elevation_list.end(),
              [](const std::pair<types::global_dof_index, double> &a,
                 const std::pair<types::global_dof_index, double> &b) {
                return a.second > b.second;
              });

    // At this point, we have a sorted list of water dof indices, sorted high to
    // low based on their elevation. We want to renumber the existing water
    // DoF indices so that they follow this ordering. We will do so
    // by making use of the fact that locally owned DoFs are numbered in
    // a contiguous block, so we can start numbering all locally owned
    // water DoFs at the first water DoF index.
    Assert(locally_owned_partitioning[0].is_contiguous(), ExcInternalError());
    Assert(locally_owned_partitioning[1].is_contiguous(), ExcInternalError());
    AssertDimension(locally_owned_partitioning[1].n_elements(),
                    water_dof_index_to_elevation_list.size());
    std::map<types::global_dof_index, types::global_dof_index>
      old_to_new_water_indices;
    if (water_dof_index_to_elevation_list.size() >
        0) // make sure this also works if a processor has no DoFs
      {
        const types::global_dof_index first_water_dof_index =
          locally_owned_partitioning[0].size() +
          *locally_owned_partitioning[1].begin();
        for (unsigned int i = 0; i < water_dof_index_to_elevation_list.size();
             ++i)
          old_to_new_water_indices[water_dof_index_to_elevation_list[i].first] =
            first_water_dof_index + i;
      }

    // Now we can do the renumbering:
    std::vector<types::global_dof_index> new_dof_numbers;
    new_dof_numbers.reserve(dof_handler.n_locally_owned_dofs());

    // Do not re-enumerate the elevation DoFs at all.
    for (const types::global_dof_index elevation_dof :
         locally_owned_partitioning[0])
      new_dof_numbers.push_back(elevation_dof);

    for (const types::global_dof_index dof_index :
         dof_handler.locally_owned_dofs())
      if (dof_index >= dof_handler.n_dofs() / 2)
        {
          Assert(old_to_new_water_indices.find(dof_index) !=
                   old_to_new_water_indices.end(),
                 ExcInternalError());
          new_dof_numbers.push_back(old_to_new_water_indices[dof_index]);
        }
    AssertDimension(new_dof_numbers.size(), dof_handler.n_locally_owned_dofs());

    dof_handler.renumber_dofs(new_dof_numbers);

    // Rebuild index sets after renumbering
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
    const types::global_dof_index n_elevation_dofs = dofs_per_block[0];
    IndexSet                      all_elevation_dofs(dof_handler.n_dofs());
    all_elevation_dofs.add_range(0, n_elevation_dofs);
    locally_relevant_water_dofs = locally_relevant_dofs;
    locally_relevant_water_dofs.subtract_set(all_elevation_dofs);
    locally_relevant_partitioning = {
      locally_relevant_dofs.get_view(0, n_elevation_dofs),
      locally_relevant_dofs.get_view(n_elevation_dofs, dof_handler.n_dofs())};

    pcout
      << "   Elevations range between "
      << Utilities::MPI::min(water_dof_index_to_elevation_list.size() > 0 ?
                               water_dof_index_to_elevation_list.back().second :
                               std::numeric_limits<double>::max(),
                             mpi_communicator)
      << "m and "
      << Utilities::MPI::max(
           water_dof_index_to_elevation_list.size() > 0 ?
             water_dof_index_to_elevation_list.front().second :
             -std::numeric_limits<double>::max(),
           mpi_communicator)
      << "m." << std::endl;
  }


  // @sect3{ParallelFlowRouter::compute_local_flow_routing()}
  //
  // Next, we need to have a function that for each degree of freedom (locally
  // owned or ghost) finds which downhill neighbor (if any) it gives water to.
  // This corresponds to the D8 scheme used in many flow routing codes, where
  // each node gives water to exactly one other node -- specifically, to the
  // neighbor in the direction of steepest descent. Because we consider the
  // four immediate neighbors of a node on a regular mesh plus the four
  // diagonal neighbors (8 neighbors total, hence "D8"), the neighbor with the
  // steepest downhill slope may not necessarily be the lowest-lying neighbor.
  //
  // The algorithm works by examining all locally relevant cells and their
  // vertices. For each vertex, we compute the slope to all other vertices in
  // the same cell. If the slope is negative (i.e., downhill), we check if it's
  // steeper than the previously found downhill direction, and if so, we record
  // it as the new destination for water from this vertex.
  //
  // Since we're working in parallel, we only have complete information about
  // the neighbors of locally owned vertices. For vertices on the boundaries
  // of the parallel partition (but still locally relevant), we need to exchange
  // information with neighboring processes via ghost cells. This is done using
  // the GridTools::exchange_cell_data_to_ghosts() function.
  //
  // The final result is stored in the local_flow_routing member variable,
  // which maps each locally relevant water degree of freedom to its downstream
  // neighbor (or to numbers::invalid_dof_index if the vertex is at a boundary
  // or in a depression with no outlet).
  void
  ParallelFlowRouter::compute_local_flow_routing()
  {
    TimerOutput::Scope t(computing_timer, "Compute local routing");
    pcout << "Computing local routing... " << std::endl;

    // First, create a map from each DoF to a pair (index,slope) of neighbors.
    // These slopes must necessarily be negative because water only flows
    // downhill.
    //
    // We initialize this map by looping over all locally relevant water DoFs
    // and setting the value of the map to (index=invalid, slope=0).
    std::map<types::global_dof_index,
             std::pair<types::global_dof_index, double>>
      water_dofs_to_steepest_downhill_neighbor_and_slope;
    for (const types::global_dof_index i : locally_relevant_water_dofs)
      water_dofs_to_steepest_downhill_neighbor_and_slope[i] = {
        numbers::invalid_dof_index, 0.0};

    // Then loop over all locally owned and ghost cells, get the locations
    // and elevations of the four vertices of each cell, along with the
    // indices of the water DoF.
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          AssertDimension(cell->reference_cell().n_vertices(), 4);

          std::array<types::global_dof_index, 4> vertex_water_dof_indices;
          std::array<Point<spacedim>, 4>         vertex_locations;
          std::array<double, 4>                  vertex_elevations;

          for (const unsigned int v : cell->reference_cell().vertex_indices())
            {
              vertex_water_dof_indices[v] = cell->vertex_dof_index(v, 1);
              vertex_locations[v]         = cell->vertex(v);

              const types::global_dof_index vertex_elevation_dof_index =
                cell->vertex_dof_index(v, 0);
              vertex_elevations[v] =
                locally_relevant_solution(vertex_elevation_dof_index);
            }

          // Next, determine the slope from each vertex to each of the other
          // vertices. If it is negative (i.e., downhill) and more negative than
          // the previously most downhill slope we have encountered for a DoF,
          // then use this as the direction in which this DoF will give water.
          // (Of course, it is possible that we encounter a steeper downhill
          // direction next on this cell, or on another cell; in that case,
          // we will simply overwrite what we determine here.)
          for (const unsigned int v : cell->reference_cell().vertex_indices())
            for (const unsigned int w : cell->reference_cell().vertex_indices())
              if (v != w)
                if (vertex_elevations[w] < vertex_elevations[v])
                  {
                    // Compute the slope between two vertices. The slope is the
                    // elevation difference divided by the distance. Since the
                    // mesh is mapped onto the surface of the Earth (a sphere),
                    // the correct
                    // distance between two points would be along the surface of
                    // the Earth. This would correctly account for the Earth's
                    // curvature and give us the true slope on the terrain. But
                    // it's
                    // also difficult to compute, so we simply use the
                    // straight-line distance between points (through the
                    // Earth), which is a good approximation for small
                    // distances.
                    const double slope =
                      (vertex_elevations[w] - vertex_elevations[v]) /
                      ((vertex_locations[v] - vertex_locations[w]).norm());
                    Assert(slope < 0, ExcInternalError());
                    if (slope <
                        water_dofs_to_steepest_downhill_neighbor_and_slope
                          [vertex_water_dof_indices[v]]
                            .second)
                      water_dofs_to_steepest_downhill_neighbor_and_slope
                        [vertex_water_dof_indices[v]] = {
                          vertex_water_dof_indices[w], slope};
                  }
        }

    // At this point, we no longer care about slopes because we have considered
    // all neighbors of all nodes and no longer need to compare slopes between
    // nodes and neighbors. So reduce the map to a smaller one that only
    // contains for each DoF who it gives water to.
    //
    // Secondly, we have worked on all locally relevant DoFs up to this point.
    // For all locally active DoFs (locally owned plus the ones on the interface
    // to ghost cells), we have considered all neighboring cells and so we can
    // be certain that we have their downstream neighbors right. But for the
    // nodes on the far side of the ghost cells (adjacent to artificial cells),
    // we have not seen all neighbor nodes, and so might have gotten wrong who
    // they give water to. As a consequence, we exclude those DoFs from the
    // reduced list that are not locally owned and instead obtain their
    // information via a ghost exchange. (We could exclude only the ones that
    // are not locally *active*, but that doesn't buy us anything and the test
    // for locally owned is cheaper because that's a contiguous set.)
    std::map<types::global_dof_index, types::global_dof_index>
      water_dofs_to_steepest_downhill_neighbor;
    for (const auto &[source_index, dest_index_and_slope] :
         water_dofs_to_steepest_downhill_neighbor_and_slope)
      if (dof_handler.locally_owned_dofs().is_element(source_index))
        water_dofs_to_steepest_downhill_neighbor.insert(
          {source_index, dest_index_and_slope.first});

    AssertDimension(water_dofs_to_steepest_downhill_neighbor.size(),
                    dof_handler.locally_owned_dofs().n_elements() / 2);

    using CellLocalData =
      std::map<types::global_dof_index, types::global_dof_index>;

    // Pack up the locally owned water dof index entries in the map
    // above for the current cell:
    const auto pack_function =
      [this, &water_dofs_to_steepest_downhill_neighbor](
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) {
        Assert(cell->is_locally_owned(), ExcInternalError());

        CellLocalData cell_local_data;
        for (const unsigned int v : cell->reference_cell().vertex_indices())
          {
            const types::global_dof_index vertex_water_dof_index =
              cell->vertex_dof_index(v, 1);
            if (dof_handler.locally_owned_dofs().is_element(
                  vertex_water_dof_index))
              {
                Assert(water_dofs_to_steepest_downhill_neighbor.find(
                         vertex_water_dof_index) !=
                         water_dofs_to_steepest_downhill_neighbor.end(),
                       ExcInternalError());
                cell_local_data.insert({vertex_water_dof_index,
                                        water_dofs_to_steepest_downhill_neighbor
                                          [vertex_water_dof_index]});
              }
          }
        return cell_local_data;
      };

    // Unpack what the other processes have sent for the current cell (which
    // is a ghost cell here). Because these were locally owned on the other
    // cell, they are necessarily not locally owned but locally relevant
    // here, and we assert that.
    //
    // We will ultimately only care about flow from one to another node if
    // at least one of them is locally active. We already know that the
    // source index is not locally active, so we discard entries that have
    // a destination that is not locally active either. (You'd think we
    // could have filtered this out in the pack_function above already, but
    // what we pack up on a cell may be sent to multiple processes that have
    // this cell as a ghost cell, and while a destination index may not be
    // locally relevant on one process, it may be on another.) We keep
    // the ones where the destination is not set, which indicates that
    // the DoF is at the boundary or in a depression without outlet.
    const auto unpack_function =
      [this, &water_dofs_to_steepest_downhill_neighbor](
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const CellLocalData &cell_local_data) {
        Assert(cell->is_ghost(), ExcInternalError());

        for (const auto &[source_index, dest_index] : cell_local_data)
          {
            Assert(dof_handler.locally_owned_dofs().is_element(source_index) ==
                     false,
                   ExcInternalError());
            Assert(locally_relevant_dofs.is_element(source_index) == true,
                   ExcInternalError());
            if ((dest_index == numbers::invalid_dof_index) ||
                locally_relevant_dofs.is_element(dest_index))
              water_dofs_to_steepest_downhill_neighbor.insert(
                {source_index, dest_index});
          }
      };

    GridTools::exchange_cell_data_to_ghosts<CellLocalData>(dof_handler,
                                                           pack_function,
                                                           unpack_function);

    // At this point, we should have gotten information about all locally
    // relevant water DoFs where they send their water (if anywhere),
    // excluding not locally owned ones that sent water to not locally
    // relevant ones -- these are at the outer fringes of the ghost layer
    // sending water further afield. In other words, we need to have
    // information about all locally active ones and at least some of
    // the locally relevant ones. We can check that this is the case:
    if constexpr (running_in_debug_mode())
      {
        Assert(water_dofs_to_steepest_downhill_neighbor.size() <=
                 locally_relevant_water_dofs.n_elements(),
               ExcInternalError());
        for (const auto &[src, dst] : water_dofs_to_steepest_downhill_neighbor)
          Assert(locally_relevant_water_dofs.is_element(src),
                 ExcInternalError());

        const types::global_dof_index n_elevation_dofs =
          dof_handler.n_dofs() / 2;
        IndexSet all_elevation_dofs(dof_handler.n_dofs());
        all_elevation_dofs.add_range(0, n_elevation_dofs);
        IndexSet locally_active_water_dofs =
          DoFTools::extract_locally_active_dofs(dof_handler);
        locally_active_water_dofs.subtract_set(all_elevation_dofs);
        for (const auto &locally_active_index : locally_active_water_dofs)
          Assert(water_dofs_to_steepest_downhill_neighbor.find(
                   locally_active_index) !=
                   water_dofs_to_steepest_downhill_neighbor.end(),
                 ExcInternalError());
      }

    // Up to this point, it was useful to work with a std::map, but ultimately
    // we want a faster representation. So convert things into a std::vector
    // of pairs:
    local_flow_routing = {water_dofs_to_steepest_downhill_neighbor.begin(),
                          water_dofs_to_steepest_downhill_neighbor.end()};

    // Finally, we can check that the only depressions we have should be
    // on the boundary of the domain. Recall that we marked depressions
    // (i.e., nodes that don't give water to any lower-lying neighbor)
    // in the src->dst relationships by invalid 'dst' values. This
    // should only be the case for 'src' nodes that are on the
    // boundary, and we can check that:
    if constexpr (running_in_debug_mode())
      {
        const IndexSet boundary_nodes =
          DoFTools::extract_boundary_dofs(dof_handler);

        for (const auto &[src, dst] : local_flow_routing)
          if (dst == numbers::invalid_dof_index)
            Assert(boundary_nodes.is_element(src),
                   ExcMessage("Found an interior depression in the DEM."));
      }
  }


  // @sect3{ParallelFlowRouter::assemble_system()}
  //
  // This function assembles the linear system that describes the steady-state
  // water flow on the landscape. The system is based on the principle of mass
  // conservation: at each point, the water flowing out must equal the water
  // flowing in (from rain and from upstream neighbors) minus any water that
  // accumulates.
  //
  // The discretized system has the form:
  //   w_i = r_i + sum_{j: j flows to i} w_j
  // where w_i is the water flow rate at node i, r_i is the rainfall at node i,
  // and the sum is over all upstream nodes j that flow into node i.
  //
  // In matrix form, this becomes:
  //   (I - F) * w = r
  // where F is the flow routing matrix (defined by the local_flow_routing
  // data), I is the identity matrix, w is the vector of water flow rates, and r
  // is the rainfall vector.
  //
  // Since we've renumbered the DoFs so that water flows from higher to lower
  // elevations, the matrix (I - F) is lower triangular, making it easy to
  // solve.
  //
  // Because we have chosen to work with a 2x2 block system where the first
  // block corresponds to elevation and the second block corresponds to water
  // flow rate, the sparsity patterns and matrix assembly are a bit more
  // complicated than in a standard finite element code, but the underlying
  // principles are the same. We just have to translate indices correctly and
  // make sure to fill the right blocks of the matrix.
  void
  ParallelFlowRouter::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "Solver 1: Assemble system");
    pcout << "Assembling linear system... " << std::endl;

    BlockDynamicSparsityPattern dsp(locally_relevant_partitioning);
    for (const types::global_dof_index water_dof_within_block_1 :
         locally_owned_partitioning[1])
      dsp.block(1, 1).add(water_dof_within_block_1, water_dof_within_block_1);

    for (const auto &[src, dst] : local_flow_routing)
      if (dof_handler.locally_owned_dofs().is_element(dst))
        dsp.add(dst, src);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    // Now fill matrix accordingly
    system_matrix.reinit(locally_owned_partitioning, dsp, mpi_communicator);
    for (const types::global_dof_index water_dof_within_block_1 :
         locally_owned_partitioning[1])
      system_matrix.block(1, 1).set(water_dof_within_block_1,
                                    water_dof_within_block_1,
                                    1.); // 1s on the diagonal
    for (const auto &[water_dof, lowest_neighbor] : local_flow_routing)
      if (dof_handler.locally_owned_dofs().is_element(lowest_neighbor))
        system_matrix.set(lowest_neighbor,
                          water_dof,
                          -1); // -1s for flow routing
    system_matrix.compress(VectorOperation::insert);

    // Then also fill rhs vector:
    const RainFallRate<spacedim> rainfall_rate;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        for (unsigned int v = 0; v < cell->reference_cell().n_vertices(); ++v)
          {
            const types::global_dof_index vertex_water_dof =
              cell->vertex_dof_index(v, 1);
            system_rhs(vertex_water_dof) +=
              rainfall_rate.value(cell->vertex(v)) * cell->measure() /
              cell->n_vertices();
          }
    system_rhs.compress(VectorOperation::add);
  }


  // @sect3{ParallelFlowRouter::FlowRoutingMatrix}
  //
  // This is a matrix-free operator class that represents the matrix A from the
  // discussion in assemble_system(). Rather than storing the matrix explicitly,
  // this class implements only the matrix-vector product (via the vmult()
  // function), computing the result on the fly from the local_flow_routing data
  // structure.
  //
  // Recall that the matrix A encodes the flow routing: A has a -1 in position
  // (i, j) if water from node j flows to node i, and 0 elsewhere, plus a +1
  // on the diagonal. Each column j has at most one non-zero entry other
  // that the diagonal entry (since each node gives water to at most
  // one downhill neighbor). More precisely, for each src->dst pair in
  // local_flow_routing, we have a -1 in position (dst, src) of the matrix.
  //
  // The vmult() function computes A*X = (I-F)*x = (I*x - F*x) by first
  // copying x to y (implementing I*x), then subtracting the contributions
  // from F. To compute F*x efficiently, we iterate over the src->dst pairs,
  // and for each one where 'dst' is in the locally owned range, we add
  // -x[src] to y[dst].
  class ParallelFlowRouter::FlowRoutingMatrix
  {
  public:
    FlowRoutingMatrix(
      const IndexSet    &locally_owned_water_dofs,
      const IndexSet    &locally_relevant_water_dofs,
      const MPI_Comm     mpi_communicator,
      const unsigned int water_dofs_offset,
      const std::vector<std::pair<types::global_dof_index,
                                  types::global_dof_index>> &local_flow_routing)
      : x_with_ghosts(locally_owned_water_dofs,
                      locally_relevant_water_dofs,
                      mpi_communicator)
      , my_local_flow_routing(local_flow_routing)
    {
      // We got the map from DoFs to downhill neighbors in global DoF
      // indices, but we need them in indices relative to the second
      // vector block (or the (1,1) matrix block). So shift, unless
      // the destination DoF is invalid, indicating that this source
      // DoF has no outlet (because it's a depression in the DEM,
      // or because it's at the boundary).
      for (auto &[src, dst] : my_local_flow_routing)
        {
          src -= water_dofs_offset;
          Assert(locally_relevant_water_dofs.is_element(src),
                 ExcInternalError());

          if (dst != numbers::invalid_dof_index)
            {
              dst -= water_dofs_offset;
              Assert(locally_relevant_water_dofs.is_element(dst),
                     ExcInternalError());
            }
        }

      // If one looks at how the vmult() function below is implemented,
      // one realizes that we only need those src->dst relationships
      // where 'dst' is a valid DoF index and is in fact in the
      // locally owned range (it is the row index in the matrix,
      // and consequently that part of the output vector we fill
      // on the current process). To make this cheaper, we erase all others
      // at this point, so we don't have to check any more there:
      Assert(locally_owned_water_dofs.is_contiguous(), ExcInternalError());
      const auto it =
        std::remove_if(my_local_flow_routing.begin(),
                       my_local_flow_routing.end(),
                       [&locally_owned_water_dofs](
                         const std::pair<types::global_dof_index,
                                         types::global_dof_index> &src_dst) {
                         const types::global_dof_index dst = src_dst.second;
                         return ((dst == numbers::invalid_dof_index) ||
                                 !locally_owned_water_dofs.is_element(dst));
                       });
      my_local_flow_routing.erase(it, my_local_flow_routing.end());

      // Pre-compute the write buffers that will be used in vmult(). These
      // buffers store the source and destination indices for efficient batch
      // operations on the vector. This approach is more efficient than looping
      // over the flow routing pairs in each vmult() call because it allows us
      // to look up many vector entries all at once, rather than having to
      // translate between global and process-local indices for each vector
      // entry we care about individually. The buffers are computed once in the
      // constructor and reused for every matrix-vector product call.
      write_buffer_source_indices.resize(my_local_flow_routing.size());
      write_buffer_indices.resize(my_local_flow_routing.size());
      write_buffer_values.resize(my_local_flow_routing.size());
      unsigned int index = 0;
      for (const auto &[src, dst] : my_local_flow_routing)
        {
          write_buffer_source_indices[index] = src;
          write_buffer_indices[index]        = dst;
          ++index;
        }
    }

    void
    vmult(typename VectorType::BlockType       &y,
          const typename VectorType::BlockType &x) const
    {
      x_with_ghosts = x;

      // The src->dst relationship defines the matrix via an entry
      // of +1 in the (src,src) position, and a -1 in the
      // (dst,src) position -- i.e., each entry in the src->dst
      // map defines a column of the matrix.
      //
      // Let us first take care of the +1s on the diagonal. We get
      // that by setting y=I*x
      y = x;

      // Then we need to add to the locally-owned elements of the y vector
      // by multiplying the x vector with the -1's of matrix.
      // This means that we need to loop over all elements of the
      // map and determine whether the row value of the entries
      // mentioned above are in the locally owned range:
      //
      // Rather than looping over each (src, dst) pair and updating y(dst)
      // individually, we use a more efficient vectorized approach via write
      // buffers. We extract all the source values at once using
      // extract_subvector_to(), negate them, and then add them to the
      // destination vector using array-based operations. This is much faster
      // than scalar operations because it allows better use of the CPU's
      // vectorization capabilities. The equivalent but slower code would be:
      //        for (const auto &[src, dst] : my_local_flow_routing)
      //        	y(dst) -= x_with_ghosts(src);
      x_with_ghosts.extract_subvector_to(write_buffer_source_indices,
                                         write_buffer_values);
      for (auto &v : write_buffer_values)
        v = -v;
      y.add(write_buffer_indices, write_buffer_values);

      y.compress(VectorOperation::add);
    }

  private:
    mutable typename VectorType::BlockType       x_with_ghosts;
    mutable std::vector<types::global_dof_index> write_buffer_source_indices;
    mutable std::vector<types::global_dof_index> write_buffer_indices;
    mutable std::vector<PetscScalar>             write_buffer_values;

    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      my_local_flow_routing;
  };


  // @sect3{ParallelFlowRouter::FlowRoutingPreconditioner}
  //
  // This class implements an efficient preconditioner for the matrix A=(I-F).
  // Since the matrix is triangular (after reordering DoFs from high to low
  // elevation), the preconditioner uses a triangular solve to approximate
  // the inverse of (I-F).
  //
  // The triangular solve works by processing the DoFs in order. For each
  // equation i (corresponding to water flow at node i), we compute:
  //   y_i = (x_i + y_i) / a_ii
  // where a_ii = 1 (the diagonal entries of (I-F) are all 1), and y_i
  // accumulates contributions from upstream nodes. We then update downstream
  // equations by adding y_i to their y values.
  //
  // Because the matrix is triangular and we process nodes in order from high
  // to low elevation, each node's solution depends only on upstream (higher)
  // nodes, which have already been processed. This makes the triangular solve
  // very efficient and cache-friendly.
  //
  // See the detailed comments in the vmult() function below for a complete
  // explanation of how the triangular solve is implemented.
  class ParallelFlowRouter::FlowRoutingPreconditioner
  {
  public:
    FlowRoutingPreconditioner(
      const IndexSet    &locally_owned_water_dofs,
      const IndexSet    &locally_relevant_water_dofs,
      const unsigned int water_dofs_offset,
      const std::vector<std::pair<types::global_dof_index,
                                  types::global_dof_index>> &local_flow_routing)
      : my_local_flow_routing(local_flow_routing)
    {
      // We got the map from DoFs to downhill neighbors in global DoF
      // indices, but we need them in indices relative to the second
      // vector block (or the (1,1) matrix block). So shift, unless
      // the destination DoF is -1, indicating that this source
      // DoF has no outlet (because it's a depression in the DEM,
      // or because it's at the boundary).
      for (auto &[src, dst] : my_local_flow_routing)
        {
          src -= water_dofs_offset;
          Assert(locally_relevant_water_dofs.is_element(src),
                 ExcInternalError());

          if (dst != numbers::invalid_dof_index)
            {
              dst -= water_dofs_offset;
              Assert(locally_relevant_water_dofs.is_element(dst),
                     ExcInternalError());
            }
        }

      // Unlike the matrix itself, the preconditioner only looks at
      // the diagonal blocks. Recall that for each local routing src->dst,
      // we have entries in the (src,src) and (dst,src) position. Both
      // of these are in the same column, 'src'. One of the two entries
      // is the diagonal entry.
      //
      // This means that for a local routing to affect the diagonal block,
      // we have to have 'src' be locally owned. That's enough: if so,
      // at least the diagonal entry (and perhaps also the other one) is
      // in the locally owned diagonal block of the matrix.
      //
      // So erase all others so that we don't have to check this during
      // the vmult() operation:
      Assert(locally_owned_water_dofs.is_contiguous(), ExcInternalError());
      const auto it =
        std::remove_if(my_local_flow_routing.begin(),
                       my_local_flow_routing.end(),
                       [&locally_owned_water_dofs](
                         const std::pair<types::global_dof_index,
                                         types::global_dof_index> &src_dst) {
                         const types::global_dof_index src = src_dst.first;
                         return !locally_owned_water_dofs.is_element(src);
                       });
      my_local_flow_routing.erase(it, my_local_flow_routing.end());

      // At this point, we should have one routing for each locally owned
      // DoF. Check that the number is right:
      AssertDimension(my_local_flow_routing.size(),
                      locally_owned_water_dofs.n_elements());

      // One last step: if we have a src->dst pair where we have already
      // made sure that 'src' is locally owned, then we know that the
      // (src,src) entry that results is in the locally owned diagonal
      // block. The second matrix entry is (dst,src), which may or may
      // not be in that diagonal block, depending on whether 'dst' is
      // locally owned. Of course, 'dst' may also be numbers::invalid_dof_index
      // if 'src' simply has no downstream neighbor.
      // In the first case, the (dst,src) matrix entry is of no concern to
      // us. In the second case, there simply is no second matrix entry.
      // In other words, in neither case is there an entry (dst,src) that
      // we need to deal with.
      //
      // To make our work easier, we turn the first into the second case
      // so that in the vmult() function we need not test for inclusion
      // of 'dst' in the index set of locally owned DoFs, but just compare
      // with numbers::invalid_dof_index.
      for (auto &src_dst : my_local_flow_routing)
        if ((src_dst.second != numbers::invalid_dof_index) &&
            (locally_owned_water_dofs.is_element(src_dst.second) == false))
          src_dst.second = numbers::invalid_dof_index;
    }


    /**
     * This function does a triangular solve with the diagonal block of
     * the matrix that corresponds to the locally owned DoFs. To see how
     * this works, consider just the following 2x2 linear system where we
     * want to solve for y given x:
     *   [a11 0  ] [y1]  =  [x1]
     *   [a21 a22] [y2]     [x2]
     * Here, we can compute y1 = x1/a11. Then we use y1 in the second
     * equation to compute
     *   y2 = (x2 - a21*y1)/a22.
     * For larger matrices, we could then compute
     *   yk = (xk - ak1*y1 - ak2*y2 - ... - a_{k,k-1}*y_{k-1}) / akk
     * The issue with this, however, is that this traverses the k'th row
     * left to right. This is appropriate if we store the matrix in a
     * row-wise storage format, as one typically does for sparse matrices
     * using the compressed row storage (CRS) format. However, here we
     * don't do that: we store information by column; specifically, we
     * store src->dst pairs which correspond to two entries both in
     * the 'src' column. In other words, it is quite difficult to
     * compute the terms
     *   - ak1*y1 - ak2*y2 - ... - a_{k,k-1}*y_{k-1}
     * because we'd have to go through all of the first k-1 src->dst
     * pairs and see whether they produce an entry in the k'th row
     * of the matrix -- where of course most don't. In any case, this would
     * lead to an O(N^2) algorithm.
     *
     * But we can do this differently by working column based. To this
     * end, let us first set y=0. Then compute
     *   y1 = x1/a11
     * and then we need to store all of the places where y1 is used
     * in subsequent equations -- in other words, we go down the first
     * column and already bring things to the right hand side. Ideally,
     * we would want to modify the system to the form
     *   [a11 0  ] [y1]  =  [x1]
     *   [0   a22] [y2]     [x2 - a21*y1]
     * where of course we would not actually modify the matrix, but simply
     * ignore the entries of the first column when solving for y2. This
     * modification requires us to write into the right hand side, which
     * we receive as a const vector, so we can't quite do that. But we
     * can store -a21*y1 in y2, which is unused at this time. Only when
     * solving the second equation, do we then solve
     *   y2 <- (x2+y2)/a22
     * where of course the right hand side is evaluated first (using what
     * was stored in y2) before the assignment to y2 happens (overwriting
     * what we had stored in it).
     *
     * For larger matrices, once we have solved for yk, we simply go
     * down the list of entries below a_{kk} and multiply these with yk
     * to add a_{lk}*yk in y_l, l>k. We then just need to solve
     *   y_l  <-  (x_l+y_l)/a_ll
     * This works for general triangular matrices. But it is particularly
     * simple here because walking down the k'th column, we *know* that
     * there is only one other entry in this column. (In fact, at most
     * one other entry: the k'th DoF may not have a downstream neighbor
     * among the locally owned DoFs.) As a consequence, once we've computed
     * yk, we have to update at most one other yl.
     *
     * As a final point, note that the diagonal entries we divide by are
     * all equal to +1. So we can just elide the division: For each k,
     * we simply have to do
     *   yk  <-  xk + yk
     * and then update downstream yl's. Moreover, in that downstream
     * update, we have to multiply by a_lk, but these are all -1s. So
     * the update is
     *   yl  <-  y_l - a_lk*y_k  = y_l + yk
     */
    void
    vmult(typename VectorType::BlockType       &y,
          const typename VectorType::BlockType &x) const
    {
      y = 0;
      for (const auto &[src, dst] : my_local_flow_routing)
        {
          // Solve the 'src'th equation. In the notation from
          // above, this reads as
          //   yk  <-  xk + yk
          // which with this function's variable names translates to
          // the following, storing the result in a temporary variable
          // for use below:
          const double yk = (y(src) += x(src));

          // Update a downstream entry if necessary. Again, in the notation
          // from above, this reads as
          //   yl  <-  yl + yk
          // and so is the following:
          if (dst != numbers::invalid_dof_index)
            y(dst) += yk;
        }
      y.compress(VectorOperation::add);
    }

  private:
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      my_local_flow_routing;
  };


  // @sect3{ParallelFlowRouter::IplusminusXMatrixBase}
  //
  // This is a base class for matrix-free operators representing matrices of the
  // form (I ± X), where X is related to the flow routing matrix.
  //
  // The class uses a block partitioning of the matrix and DoFs:
  // - Diagonal block (locally owned rows and columns)
  // - R matrix (locally owned rows, but columns on other processes)
  //
  // The vmult() function computes y=(I ± X)*x by splitting the computation into
  // contributions from the locally owned part and the off-process part.
  //
  // The derived classes IplusXMatrix and IminusXMatrix implement the two
  // variants of the actual operator.
  class ParallelFlowRouter::IplusminusXMatrixBase
  {
  public:
    IplusminusXMatrixBase(
      const IndexSet    &locally_owned_water_dofs,
      const IndexSet    &locally_relevant_water_dofs,
      const MPI_Comm     mpi_communicator,
      const unsigned int water_dofs_offset,
      const std::vector<std::pair<types::global_dof_index,
                                  types::global_dof_index>> &local_flow_routing,
      const FlowRoutingPreconditioner &flow_routing_preconditioner)
      : tmp(locally_owned_water_dofs, mpi_communicator)
      , x_with_ghosts(locally_owned_water_dofs,
                      locally_relevant_water_dofs,
                      mpi_communicator)
      , my_local_flow_routing(local_flow_routing)
      , flow_routing_preconditioner(flow_routing_preconditioner)
    {
      // We got the map from DoFs to downhill neighbors in global DoF
      // indices, but we need them in indices relative to the second
      // vector block (or the (1,1) matrix block). So shift, unless
      // the destination DoF is -1, indicating that this source
      // DoF has no outlet (because it's a depression in the DEM,
      // or because it's at the boundary).
      for (auto &[src, dst] : my_local_flow_routing)
        {
          src -= water_dofs_offset;
          Assert(locally_relevant_water_dofs.is_element(src),
                 ExcInternalError());

          if (dst != numbers::invalid_dof_index)
            {
              dst -= water_dofs_offset;
              Assert(locally_relevant_water_dofs.is_element(dst),
                     ExcInternalError());
            }
        }

      // Compared to the vmult() function of the FlowRoutingMatrix, where
      // we needed all entries that are in the locally owned rows of the
      // matrix, for the current matrix all we need are those entries that
      // are in the locally owned rows *but not in the locally owned columns*.
      // As a consequence, delete not only everything that's not in locally
      // owned rows, but *also* those *are* in locally owned columns:
      Assert(locally_owned_water_dofs.is_contiguous(), ExcInternalError());
      const auto it =
        std::remove_if(my_local_flow_routing.begin(),
                       my_local_flow_routing.end(),
                       [&locally_owned_water_dofs](
                         const std::pair<types::global_dof_index,
                                         types::global_dof_index> &src_dst) {
                         const types::global_dof_index from = src_dst.first;
                         const types::global_dof_index to   = src_dst.second;
                         return (locally_owned_water_dofs.is_element(from) ||
                                 ((to == numbers::invalid_dof_index) ||
                                  !locally_owned_water_dofs.is_element(to)));
                       });
      my_local_flow_routing.erase(it, my_local_flow_routing.end());

      // On a single process, the B matrix is empty: We have only one diagonal
      // block that spans the whole matrix, and so there is literally nothing
      // left for B. Make sure that is in fact true.
      Assert((Utilities::MPI::n_mpi_processes(mpi_communicator) > 1) ||
               (my_local_flow_routing.size() == 0),
             ExcInternalError());

      // Pre-compute the write buffers for efficient batch vector operations
      // in the vmult() function. This approach avoids repeated allocations and
      // allows vectorized extraction and addition operations on the flow
      // routing pairs, which is significantly faster than processing them
      // one-by-one.
      write_buffer_source_indices.resize(my_local_flow_routing.size());
      write_buffer_indices.resize(my_local_flow_routing.size());
      write_buffer_values.resize(my_local_flow_routing.size());
      unsigned int index = 0;
      for (const auto &[src, dst] : my_local_flow_routing)
        {
          write_buffer_source_indices[index] = src;
          write_buffer_indices[index]        = dst;
          ++index;
        }
    }

  protected:
    mutable typename VectorType::BlockType       tmp;
    mutable typename VectorType::BlockType       x_with_ghosts;
    mutable std::vector<types::global_dof_index> write_buffer_source_indices;
    mutable std::vector<types::global_dof_index> write_buffer_indices;
    mutable std::vector<PetscScalar>             write_buffer_values;

    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                     my_local_flow_routing;
    const FlowRoutingPreconditioner &flow_routing_preconditioner;
  };


  // @sect3{ParallelFlowRouter::IplusXMatrix}
  //
  // This class represents the matrix (I + X), where X is derived from the
  // flow routing matrix. It is used in certain implicit time-stepping schemes.
  // The class simply inherits from IplusminusXMatrixBase.
  class ParallelFlowRouter::IplusXMatrix
    : public ParallelFlowRouter::IplusminusXMatrixBase
  {
  public:
    IplusXMatrix(const IndexSet    &locally_owned_water_dofs,
                 const IndexSet    &locally_relevant_water_dofs,
                 const MPI_Comm     mpi_communicator,
                 const unsigned int water_dofs_offset,
                 const std::vector<
                   std::pair<types::global_dof_index, types::global_dof_index>>
                                                 &local_flow_routing,
                 const FlowRoutingPreconditioner &flow_routing_preconditioner)
      : IplusminusXMatrixBase(locally_owned_water_dofs,
                              locally_relevant_water_dofs,
                              mpi_communicator,
                              water_dofs_offset,
                              local_flow_routing,
                              flow_routing_preconditioner)
    {}


    void
    vmult(typename VectorType::BlockType       &y,
          const typename VectorType::BlockType &x) const
    {
      // Start by importing ghost entries:
      x_with_ghosts = x;

      // The from->to relationship defines the matrix via an entry
      // of +1 in the (from,from) position, and a -1 in the
      // (to,from) position -- i.e., each entry in the from->to
      // map defines a column of the matrix.
      //
      // The +1s all lie outside the D matrix, so we need not think
      // about these entries at all, and we can start with a zero vector:
      tmp = 0;

      // Then we need to add to the locally-owned elements of the dst vector
      // by multiplying the src vector with the -1's of matrix.
      // This means that we need to loop over all elements of the
      // map and determine whether the row value of the entries
      // mentioned above are in the locally owned range:

      // We use the write buffer optimization to efficiently compute the
      // matrix-vector product with vectorized batch operations. The code below
      // is equivalent to:
      //      const std::pair<types::global_dof_index, types::global_dof_index>
      //        locally_owned_range = tmp.local_range();
      //      for (const auto &[src, dst] : my_local_flow_routing)
      //        {
      //          Assert((dst != numbers::invalid_dof_index) &&
      //                   (dst >= locally_owned_range.first) &&
      //                   (dst < locally_owned_range.second),
      //                 ExcInternalError());
      //          tmp(dst) -= x_with_ghosts(src);
      //        }
      // However, this version is much more efficient because it uses array
      // operations that can be vectorized by the CPU, rather than scalar
      // operations in a loop.
      x_with_ghosts.extract_subvector_to(write_buffer_source_indices,
                                         write_buffer_values);
      for (auto &v : write_buffer_values)
        v = -v;
      tmp.add(write_buffer_indices, write_buffer_values);

      tmp.compress(VectorOperation::add);

      // We have now computed tmp=R*x. Let's apply D^{-1} to it
      // to get X*x:
      flow_routing_preconditioner.vmult(y, tmp);

      // Finally, we need to add to it src so that we get (I+X)*src:
      y += x;
    }
  };


  // @sect3{ParallelFlowRouter::IminusXMatrix}
  //
  // This class represents the matrix (I - X), where X is derived from the
  // flow routing matrix. The implementation is very similar to IplusXMatrix,
  // but with the opposite sign for X.
  class ParallelFlowRouter::IminusXMatrix
    : public ParallelFlowRouter::IplusminusXMatrixBase
  {
  public:
    IminusXMatrix(const IndexSet    &locally_owned_water_dofs,
                  const IndexSet    &locally_relevant_water_dofs,
                  const MPI_Comm     mpi_communicator,
                  const unsigned int water_dofs_offset,
                  const std::vector<
                    std::pair<types::global_dof_index, types::global_dof_index>>
                                                  &local_flow_routing,
                  const FlowRoutingPreconditioner &flow_routing_preconditioner)
      : IplusminusXMatrixBase(locally_owned_water_dofs,
                              locally_relevant_water_dofs,
                              mpi_communicator,
                              water_dofs_offset,
                              local_flow_routing,
                              flow_routing_preconditioner)
    {}


    void
    vmult(typename VectorType::BlockType       &y,
          const typename VectorType::BlockType &x) const
    {
      x_with_ghosts = x;

      tmp = 0;
      x_with_ghosts.extract_subvector_to(write_buffer_source_indices,
                                         write_buffer_values);
      tmp.add(write_buffer_indices, write_buffer_values);

      tmp.compress(VectorOperation::add);

      // We have now computed tmp=-R*x. Let's apply D^{-1} to it
      // to get -X*x:
      flow_routing_preconditioner.vmult(y, tmp);

      // Finally, we need to add to it src so that we get (I-X)*src:
      y += x;
    }
  };


  // @sect3{ParallelFlowRouter::assemble_matrix_free_operators()}
  //
  // Rather than assembling the full matrix (which could be very large in
  // parallel), this function creates "matrix-free" operators that compute
  // matrix-vector products implicitly. We will then be able to compare
  // all of these approaches.
  //
  // Specifically, this function creates three key objects:
  // 1. A FlowRoutingMatrix that represents the matrix F described in
  //    assemble_system() above.
  // 2. A FlowRoutingPreconditioner that approximates the inverse of (I - F)
  //    using a fast triangular solve.
  // 3. A pair of matrices I+X and I-X.
  //
  // These matrix-free operators are used in the solve() function to perform
  // iterative linear solves without ever explicitly storing the full matrix.
  void
  ParallelFlowRouter::assemble_matrix_free_operators()
  {
    TimerOutput::Scope t(computing_timer,
                         "Solver 2: Assemble matrix-free operators");
    pcout << "Assembling matrix-free operators... " << std::endl;

    flow_routing_matrix = std::make_unique<const FlowRoutingMatrix>(
      locally_owned_partitioning[1],
      locally_relevant_partitioning[1],
      mpi_communicator,
      locally_relevant_partitioning[0].size(),
      local_flow_routing);

    flow_routing_preconditioner =
      std::make_unique<const FlowRoutingPreconditioner>(
        locally_owned_partitioning[1],
        locally_relevant_partitioning[1],
        locally_relevant_partitioning[0].size(),
        local_flow_routing);

    I_plus_X_matrix = std::make_unique<const IplusXMatrix>(
      locally_owned_partitioning[1],
      locally_relevant_partitioning[1],
      mpi_communicator,
      locally_relevant_partitioning[0].size(),
      local_flow_routing,
      *flow_routing_preconditioner);

    I_minus_X_matrix = std::make_unique<const IminusXMatrix>(
      locally_owned_partitioning[1],
      locally_relevant_partitioning[1],
      mpi_communicator,
      locally_relevant_partitioning[0].size(),
      local_flow_routing,
      *flow_routing_preconditioner);
  }



  // @sect3{ParallelFlowRouter::solve()}
  //
  // This function solves the linear system assembled in assemble_system() to
  // find the steady-state water flow rates at all nodes on the mesh. Since we
  // have renumbered the degrees of freedom from high to low elevation, the
  // system matrix is lower triangular, and we can solve it very efficiently.
  //
  // The function uses an iterative Richardson solver with a preconditioner
  // derived from the triangular structure of the matrix. Because the matrix
  // is triangular, a simple SOR preconditioner with a relaxation factor of
  // 1.0 is equivalent to the triangular solve.
  //
  // The function then solves the system three more times using the matrix-free
  // operators defined above, and compares the results to verify that they all
  // give the same solution.
  void
  ParallelFlowRouter::solve()
  {
    pcout << "Solving for global water routing... " << std::endl;

    // ---------------- Solve matrix-based -------------------------
    VectorType completely_distributed_solution_matrix_based(
      locally_owned_partitioning, mpi_communicator);
    {
      TimerOutput::Scope t(computing_timer,
                           "Solver 1: Solve for water matrix-based");

      SolverControl solver_control(dof_handler.n_dofs() / 2,
                                   1e-6 * system_rhs.block(1).l2_norm());
      SolverRichardson<typename VectorType::BlockType> solver(solver_control);

      PETScWrappers::PreconditionSOR preconditioner;
      preconditioner.initialize(system_matrix.block(1, 1));

      solver.solve(system_matrix.block(1, 1),
                   completely_distributed_solution_matrix_based.block(1),
                   system_rhs.block(1),
                   preconditioner);

      pcout << "   Solved matrix-based in " << solver_control.last_step()
            << " iterations." << std::endl;
    }


    // ---------------- Solve matrix-free -------------------------
    VectorType completely_distributed_solution_matrix_free(
      locally_owned_partitioning, mpi_communicator);
    {
      TimerOutput::Scope t(computing_timer,
                           "Solver 2: Solve for water matrix-free");

      SolverControl solver_control(dof_handler.n_dofs() / 2,
                                   1e-6 * system_rhs.block(1).l2_norm());
      SolverRichardson<typename VectorType::BlockType> solver(solver_control);

      solver.solve(*flow_routing_matrix,
                   completely_distributed_solution_matrix_free.block(1),
                   system_rhs.block(1),
                   *flow_routing_preconditioner);

      pcout << "   Solved matrix-free in " << solver_control.last_step()
            << " iterations." << std::endl;
    }

    // ---------------- Solve via (I+X)x=D^{-1}b  -------------------------
    VectorType completely_distributed_solution_IplusX(
      locally_owned_partitioning, mpi_communicator);
    {
      TimerOutput::Scope t(computing_timer, "Solver 3: Solve for water I+X");

      typename VectorType::BlockType Dinv_times_rhs(
        locally_owned_partitioning[1], mpi_communicator);
      flow_routing_preconditioner->vmult(Dinv_times_rhs, system_rhs.block(1));

      SolverControl solver_control(dof_handler.n_dofs() / 2,
                                   1e-6 * Dinv_times_rhs.l2_norm());
      SolverRichardson<typename VectorType::BlockType> solver(solver_control);

      solver.solve(*I_plus_X_matrix,
                   completely_distributed_solution_IplusX.block(1),
                   Dinv_times_rhs,
                   PreconditionIdentity());

      pcout << "   Solved I+X-based in " << solver_control.last_step()
            << " iterations." << std::endl;
    }

    // ---------------- Solve via (I-X)(I+X)x=(I-X)D^{-1}b -------------------
    VectorType completely_distributed_solution_IminusX_IplusX(
      locally_owned_partitioning, mpi_communicator);
    {
      TimerOutput::Scope t(computing_timer,
                           "Solver 4: Solve for water (I-X)(I+X)");

      typename VectorType::BlockType Dinv_times_rhs(
        locally_owned_partitioning[1], mpi_communicator);
      flow_routing_preconditioner->vmult(Dinv_times_rhs, system_rhs.block(1));

      SolverControl solver_control(dof_handler.n_dofs() / 2,
                                   1e-6 * Dinv_times_rhs.l2_norm());
      SolverRichardson<typename VectorType::BlockType> solver(solver_control);

      solver.solve(*I_plus_X_matrix,
                   completely_distributed_solution_IminusX_IplusX.block(1),
                   Dinv_times_rhs,
                   *I_minus_X_matrix);

      pcout << "   Solved (I-X)(I+X)-based in " << solver_control.last_step()
            << " iterations." << std::endl;
    }

    locally_relevant_solution.block(1) =
      completely_distributed_solution_matrix_free.block(1);

    // ----------- Now make sure the solutions agree:
    completely_distributed_solution_matrix_free -=
      completely_distributed_solution_matrix_based;
    pcout << "   Relative error between matrix-based and matrix-free: "
          << completely_distributed_solution_matrix_free.l2_norm() /
               completely_distributed_solution_matrix_based.l2_norm()
          << std::endl;

    completely_distributed_solution_IplusX -=
      completely_distributed_solution_matrix_based;
    pcout << "   Relative error between matrix-based and I+X solution: "
          << completely_distributed_solution_IplusX.l2_norm() /
               completely_distributed_solution_matrix_based.l2_norm()
          << std::endl;

    completely_distributed_solution_IminusX_IplusX -=
      completely_distributed_solution_matrix_based;
    pcout << "   Relative error between matrix-based and (I-X)(I+X) solution: "
          << completely_distributed_solution_IminusX_IplusX.l2_norm() /
               completely_distributed_solution_matrix_based.l2_norm()
          << std::endl;
  }


  // @sect3{ParallelFlowRouter::check_conservation_for_waterflow_system()}
  //
  // This function verifies that the solution satisfies the principle of mass
  // conservation for water. Specifically, it checks that the total water input
  // (from rainfall) equals the total water output (flowing out of the domain).
  //
  // The function computes:
  // - The total water input by integrating the rainfall rate over the domain
  // - The total water output by summing the water flow rates at boundary nodes
  //
  // If these two quantities differ by more than a small tolerance (currently
  // 1%), the function throws an exception, indicating a problem with the
  // solution.
  //
  // This is a valuable diagnostic check: in a correctly formulated and solved
  // system, mass should be strictly conserved (up to numerical errors). If
  // conservation is violated, it indicates an error in problem setup, assembly,
  // or solution.
  void
  ParallelFlowRouter::check_conservation_for_waterflow_system(
    const VectorType &solution)
  {
    TimerOutput::Scope t(computing_timer, "Water conservation check");

    const QGauss<dim>       quadrature_formula(fe.degree + 1);
    FEValues<dim, spacedim> fe_values(fe,
                                      quadrature_formula,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);

    const RainFallRate<spacedim> rainfall_rate;
    double                       input_from_rain_rate = 0.0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            input_from_rain_rate +=
              rainfall_rate.value(fe_values.quadrature_point(q)) *
              fe_values.JxW(q);
        }

    input_from_rain_rate =
      Utilities::MPI::sum(input_from_rain_rate, mpi_communicator);


    // Now also check the outflow. Water flows out of the domain at points
    // that (i) are at the boundary, and (ii) have no lower neighbors. We
    // have built the DEM so that it has no local depressions in the
    // interior of the domain, so we only have to check (ii). In the
    // local water routing table, this is indicated by (src->dst) pairs
    // where 'dst' is an invalid_dof_index. The only thing we have to pay
    // attention to is that we only count the locally owned DoFs to
    // avoid double-counting:
    double water_outflow_rate = 0;
    for (const auto &[src, dst] : local_flow_routing)
      if ((dst == numbers::invalid_dof_index) &&
          dof_handler.locally_owned_dofs().is_element(src))
        water_outflow_rate += solution(src);

    water_outflow_rate =
      Utilities::MPI::sum(water_outflow_rate, mpi_communicator);

    const double error_abs =
      std::abs(input_from_rain_rate - water_outflow_rate);
    const double error_rel = std::abs(error_abs / input_from_rain_rate);
    pcout << "Conservation check (water)" << std::endl
          << "   Input:          " << input_from_rain_rate << std::endl
          << "   Output:         " << water_outflow_rate << std::endl
          << "   Relative error: " << error_rel << std::endl;

    AssertThrow(error_rel < 1e-2,
                ExcMessage("Conservation of water rate not satisfied."));
  }


  // @sect3{ParallelFlowRouter::output_results()}
  //
  // This function writes the solution to VTU output files for visualization.
  // The function also adds the subdomain ID of each cell, which is useful for
  // visualizing the parallel partitioning of the mesh across MPI processes.
  // This can help verify that the partitioning is reasonably balanced.
  //
  // In parallel, each process writes its local portion of the solution to a
  // separate file, and a master PVTU file is created that ties all the pieces
  // together for visualization in tools like ParaView or VisIt.
  void
  ParallelFlowRouter::output_results()
  {
    TimerOutput::Scope t(computing_timer, "Output");
    pcout << "Writing output... " << std::flush;

    const std::vector<std::string> solution_names = {"elevation",
                                                     "water_flow_rate"};
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation = {
        DataComponentInterpretation::component_is_scalar,
        DataComponentInterpretation::component_is_scalar};

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim, spacedim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_communicator, 2);
  }



  // @sect3{ParallelFlowRouter::run()}
  //
  // This is the main entry point for the flow routing solver. It orchestrates
  // all the steps needed to solve the water flow routing problem:
  //
  // 1. Create the computational mesh (`make_grid()`)
  // 2. Set up the finite element spaces and DoF numbering (`setup_dofs()`)
  // 3. Interpolate the digital elevation model onto the mesh
  // (`interpolate_initial_elevation()`)
  // 4. Renumber DoFs so water flows from high to low elevation
  // (`sort_dofs_high_to_low()`)
  // 5. Determine which downhill neighbor each node flows to
  // (`compute_local_flow_routing()`)
  // 6. Assemble the linear system for steady-state flow (`assemble_system()`)
  // 7. Set up matrix-free operators for efficient solving
  // (`assemble_matrix_free_operators()`)
  // 8. Solve the linear system (`solve()`)
  // 9. Check conservation of water mass
  // (`check_conservation_for_waterflow_system()`)
  // 10. Write output for visualization (`output_results()`)
  // 11. Print performance statistics (`computing_timer.print_summary()`)
  //
  // The sequence of these steps reflects the logical flow of the algorithm:
  // we first set up the geometry and DoFs, then compute the flow routing
  // connectivity, then assemble and solve the linear system, and finally
  // perform validation and output.
  void
  ParallelFlowRouter::run()
  {
    make_grid();
    setup_dofs();
    interpolate_initial_elevation();
    sort_dofs_high_to_low();

    compute_local_flow_routing();

    assemble_system();
    assemble_matrix_free_operators();

    solve();

    check_conservation_for_waterflow_system(locally_relevant_solution);
    if (generate_graphical_output)
      output_results();

    // Print the time taken for each section of the code, first in the summary
    // table and then as individual numbers in one line. This is useful for
    // creating graphs of run times.
    computing_timer.print_summary();
    pcout << "Times per section: ";
    for (const auto &[name, time] :
         computing_timer.get_summary_data(TimerOutput::total_wall_time))
      pcout << time << ' ';
    pcout << std::endl;
  }
} // namespace ParallelFlowRouting



// @sect3{The main() function}
//
// The main function of the program is quite simple. It initializes MPI for
// parallel execution, creates a ParallelFlowRouter object that registers all
// necessary parameters, and then runs the flow routing algorithm.
//
// The parameters can be provided in a file (by passing the filename as a
// command-line argument) or are left at their defaults.
//
// The function includes basic error handling: if an exception occurs during
// the computation, it prints an error message and exits gracefully with a
// non-zero return code.
int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Create the problem object (this registers parameters with
      // ParameterAcceptor)
      ParallelFlowRouting::ParallelFlowRouter problem;

      // Parse parameters from file if provided
      if (argc > 1)
        ParameterAcceptor::initialize(argv[1]);
      else
        ParameterAcceptor::initialize();

      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
