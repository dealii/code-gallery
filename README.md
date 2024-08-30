# code-gallery
The [code gallery](https://dealii.org/developer/doxygen/deal.II/CodeGallery.html)
is a collection of codes based on [deal.II](https://www.dealii.org)
contributed by deal.II users. These codes typically solve problems more
complicated than those in the deal.II tutorial. Their intention is to solve
actual problems, rather than demonstrate particular aspects of deal.II. The
code gallery is, however, cross linked from the
[set of tutorials](https://dealii.org/developer/doxygen/deal.II/Tutorial.html#TutorialConnectionGraph).

### Building the documentation

To build the doxygen documentation, simply check out the code gallery git
repository into the main deal.II source directory (parallel to the
`examples`, or `doc` directory), and then build the deal.II documentation
as
[described in the readme](https://www.dealii.org/developer/readme.html#documentation).
It will pick up the code gallery and create joint documentation for the
tutorial and the code gallery.

### Maintenance of contributed codes

The examples in the code-gallery are periodically adjusted so that they
maintain compatibility with recent versions of deal.II. This means
that their implementation may be modified slightly in the process, perhaps breaking
compatibility with older versions of deal.II. Older version of each
contributed code may be attained by checking out the appropriate git commit
of the code-gallery repository.

## Contributing to the gallery
First: **We appreciate all contributions!** 
We have tried to set the bar to entry as low as possible.
In essence, here is all you have to do:

1. Create a fork of this repository. 
   For this, you need to have a GitHub account. 
   Log in, then go to this repository again and click on the `Fork` symbol at the top right.
2. This yields a copy of the code gallery repository 
   to which you are allowed to write (a 'fork').
   At the right side of the page, you will find its HTTPS address. 
   Check out a copy of it to your local hard drive via
   ```
    git clone https://github.com/yourusername/code-gallery.git
   ```
   where you replace `myusername` with your GitHub user name.
3. Go into the `code-gallery` directory that was just generated.
   Within it, create a new directory for your project, named as you like. 
4. Create and switch to a branch in your local git copy:
   ```
   git branch my-code-gallery-project
   git checkout my-code-gallery-project
   ```
5. Put your source files and everything else you need to build 
   and run the code into this directory. 
   In addition, the following files and subdirectories need to exist:
   * `Readme.md:` A [markdown formatted](https://daringfireball.net/projects/markdown/basics) file that provides basic overview over what this program does. You can use LaTeX-style formulas include in `$...$`, or offset via
     ```
     @f{align*}{
      ...
     @f}
     ```
     to explain ideas of the program.
   * `CMakeLists.txt:` A CMake file that allows others to build your code. 
     It must be possible to simply run it via
     ```
     cmake -DDEAL_II_DIR=/path/to/dealii . ; make ; make run
     ```
   * `doc/entry-name:` A simple text file that contains a simple, short name of the program. 
     This will likely match the directory name, 
     but can contain spaces and special characters (encoded in HTML, if necessary). 
     This name will be used in the title line of the program's page, 
     as well as in the [list of programs](https://dealii.org/developer/doxygen/deal.II/CodeGallery.html).
   * `doc/tooltip:` A one-line description of the program that is shown when hovering over a symbol in the connection graph.
   * `doc/build-on:` A textfile containing a list of the form
     ```
     step-XX step-YY gallery-NNNN gallery-MMMM
     ```
     where `XX` and `YY` are the numbers of existing tutorial programs that your application builds on, 
     and `NNNN` and `MMMM` are the directory names of existing gallery applications. 
     This information is used to build the 
     [connection graph for tutorial and code gallery applications.](https://dealii.org/developer/doxygen/deal.II/Tutorial.html#graph)
   * `doc/author:` A text file containing the list of authors and their email adresses in the form
     ```
     Jane Doe <jane@doe.org>,
     Helga Mustermann <helga@email.de>
     ```
   * `doc/dependencies:` A file with a list of entries of the form
     ```
     DEAL_II_WITH_CXX11 DEAL_II_WITH_TRILINOS
     ```
     which specifies the requirements you need the underlying deal.II installation to meet for the program to run.
     The entries are CMake variables that are set during deal.II cofiguration
     and correspond to the items you find printed in the summary at the end of a `cmake` run when configuring deal.II
6. Add your code gallery directory (here: `my-project-name`)   to git 
   and upload it to your fork of the repository:
   ```
   git add my-project-name 
   git push -u origin my-code-gallery-project
   ```
7. Go to your GitHub code-gallery page,
   which should be named `https://github.com/myusername/code-gallery`.
   Sign in if necessary.
   It should allow you to create a 'pull request' for your new code gallery project.
   Do so -- this will alert the deal.II maintainers to the fact that you want to contribute a new project,
   and sets up a short code for the review process.
 