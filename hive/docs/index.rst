.. Hives- A P2P Stochastic Swarm Guidance Simulator documentation master file, created by
   sphinx-quickstart on Wed Aug 19 12:05:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hives - A P2P Stochastic Swarm Guidance Simulator
============================================================

.. toctree::
   :maxdepth: 4
   :caption: Navbar
   :hidden:

   quickstartdocs
   scriptdocs
   app
   notedocs
   indices

This project was born during the development and writing of my master's
dissertation in Computer Science and Engineering, through a research
sponsorship granted to me by ISR_ (ISR) to participate in the project
*UID/EEA/50009/2019 - 1801P.00920.1.02 DSOR*. ISR is a research and
development institution affiliated with IST_, the university where my
dissertation will be submitted.

My work consisted of adapting and optimizing a few algorithms widely used and
studied in robotics and control research fields to a P2P scenario where the
peers form the basis for a Distributed Backup System. Throughout this process,
I ended up writing my own cycle based simulator, and the result was
*Hives* due to the lack of other viable options.

Hives is a P2P Stochastic Swarm Guidance Simulator that facilitates
research by allowing developers to prototype P2P networks based on swarm
guidance behaviors quickly. The simulator is written in Python_ (version 3.7.7),
which offers users easy access to powerful scientific libraries such as NumPy_,
SciPy_, and Pandas_, which are not readily available in languages like Java_
and as a result of some of the best or most well-known simulators out there.

|pic1| |pic2|

.. |pic1| image:: _static/logos/ist-logo.png
   :width: 45%

.. |pic2| image:: _static/logos/isr-logo.png
   :width: 45%

.. _IST: https://tecnico.ulisboa.pt/en/

.. _ISR: https://welcome.isr.tecnico.ulisboa.pt/

.. _Python: http://www.python.org/

.. _NumPy: https://numpy.org/

.. _SciPy: https://www.scipy.org/

.. _Pandas: https://pandas.pydata.org/

.. _Java: https://www.oracle.com/java/technologies/javase-jdk14-downloads.html/
