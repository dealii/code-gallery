# code-gallery
The [code gallery](https://dealii.org/developer/doxygen/deal.II/CodeGallery.html)
is a collection of codes based on [deal.II](https://www.dealii.org)
contributed by `deal.II` users. These codes typically solve problems more
complicated than those in the deal.II tutorial. Their intention is to solve
actual problems, rather than demonstrate particular aspects of deal.II. The
code gallery is, however, cross linked from the
[set of tutorials](https://dealii.org/developer/doxygen/deal.II/Tutorial.html#TutorialConnectionGraph).

### Building the documentation

To build the doxygen documentation, simply check out the code gallery git
repository into the main deal.II source directory (parallel to the
`examples`, or `doc` directory), and then build the `deal.II` documentation
as
[described in the readme](https://www.dealii.org/developer/readme.html#documentation).
It will pick up the code gallery and create joint documentation for the
tutorial and the code gallery.

### Maintainance of contributed codes

The examples in the code-gallery of periodically adjusted so that they maintain compatibility with a "recent" version of the `deal.II`. This means that their implementation may be modified slightly in the process, breaking compatibility with older versions of `deal.II`. Older version of each contributed code may be attained by checking out the appropriate git commit of the code-gallery repository.

