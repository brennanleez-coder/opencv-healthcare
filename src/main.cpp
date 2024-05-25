#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // For scoped_interpreter
#include <iostream>

using namespace std;
namespace py = pybind11;

int main()
{
    py::scoped_interpreter guard{}; // Ensures Python interpreter is properly managed

    try
    {
        /* this approach does not activate the venv
         * it just points to the venv's site-packages, good enough for now
         */

        auto sys = py::module_::import("sys");
        auto os = py::module_::import("os");

        // Set the project root directory explicitly
        string project_root = os.attr("path").attr("join")(os.attr("getcwd")(), "..").cast<string>();
        project_root = os.attr("path").attr("abspath")(project_root).cast<string>();
        cout << "Project root directory: " << project_root << endl;

        // Path to the virtual environment, now relative to the project root
        string venv_path = project_root + "/.venv";
        string site_packages_path = venv_path + "/lib/python3.11/site-packages";
        cout << "Site packages path: " << site_packages_path << endl;

        // Update sys.path to include the virtual environment's packages and the module path
        sys.attr("path").attr("insert")(1, site_packages_path);
        sys.attr("path").attr("insert")(0, project_root + "/sit_stand_algorithm/src");
        sys.attr("path").attr("insert")(0, project_root);

        // Print the current Python path for debugging
        // cout << "Current Python sys.path:" << endl;
        // for (auto path : sys.attr("path"))
        // {
        //     cout << path.cast<string>() << endl;
        // }
        auto main_module = py::module_::import("sit_stand_algorithm.src.main");
        main_module.attr("main")();
    }
    catch (const py::error_already_set &e)
    {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
