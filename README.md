# code-gallery
The [code gallery](https://dealii.org/developer/doxygen/deal.II/CodeGallery.html) is a collection of codes based on
[deal.II](https://www.dealii.org) contributed by `deal.II`
users. These codes typically solve problems more complicated than
those in the deal.II tutorial. Their intention is to solve actual
problems, rather than demonstrate particular aspects of deal.II.
The code gallery is, however, cross linked from the
[set of tutorials](https://dealii.org/developer/doxygen/deal.II/Tutorial.html#TutorialConnectionGraph).

### Building the documentation

To build the doxygen documentation, check out the code gallery git repository in a directory parallel to the `<path/to/deal.II>/examples/` directory, and then build the `deal.II` documentation as [described in the readme](https://www.dealii.org/developer/readme.html#documentation). 
It will pick up the code gallery and create joint documentation for the tutorial and the code gallery. 
