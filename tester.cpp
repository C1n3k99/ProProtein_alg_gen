#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct {
    std::string program_file;
    std::string results_file;
    std::string params_file;
    bool is_kv = false;
} config;

std::ofstream results_file;

std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&currentTime), "%d-%m-%Y %H:%M:%S");
    return ss.str();
}

void runTester(const std::vector<std::string> &params) {
    std::cout << ">";
    for (const auto& param : params) {
        std::cout << " " << param;
    }
    std::cout << std::endl;

    // Construct the command to execute
    std::string command = config.program_file;
    for (const auto& param : params) {
        command += " " + param;
    }

    // Execute the program
    std::system(command.c_str());
}

void runProgram(const std::vector<std::string> &params) {
    std::cout << "Command: " << config.program_file;
    for (const auto &param : params) {
        std::cout << " " << param;
    }
    std::cout << std::endl;

    // Construct the command to execute
    std::string command = config.program_file;
    for (const auto &param : params) {
        command += " " + param;
    }

    // Measure the execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string timestampStart = getCurrentDateTime();

    // Open a pipe to capture the output of the command
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error executing command!" << std::endl;
        return;
    }
    // Redirect the output of the command to the log file
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string timestampEnd = getCurrentDateTime();
        std::string tester_fields = timestampStart + ";" + timestampEnd + ";" + command + ";";
        results_file << tester_fields << buffer;
        results_file.flush();
        std::cout << "[" << timestampEnd << "] " << buffer;
        std::cout.flush();
        timestampStart = getCurrentDateTime();
    }
    // Calculate and display execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl << std::endl;

    // Check the status of the command
    int status = pclose(pipe);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        std::cerr << "Command execution failed with status: " << WEXITSTATUS(status) << std::endl;
        return;
    }
}

// Function to trim leading and trailing whitespaces from a string
void trim(std::string &str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), str.end());
}

void printParams(const std::vector<std::pair<std::string, std::vector<std::string>>> &params) {
    std::cout << "Params:" << std::endl;
    for (const std::pair<std::string, std::vector<std::string>> &param : params) {
        std::cout << " [" << param.first << "]" << std::endl;
        for (const std::string &arg : param.second) {
            std::cout << "  [" << arg << "]" << std::endl;
        }
    }
}

// Function to parse YAML file into a vector of pairs
std::vector<std::pair<std::string, std::vector<std::string>>> parseYAML(const std::string &filename) {
    std::ifstream input(filename);

    std::vector<std::pair<std::string, std::vector<std::string>>> params;
    std::pair<std::string, std::vector<std::string>> arg;

    std::string line;

    while (std::getline(input, line)) {
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        trim(line);
        if (line.empty()) {
            continue;
        }
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos && colon_pos != line.size() - 1) {
            // found ":" and it's in the middle, so it's one line parameter "key: value"
            params.push_back(arg);
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            trim(key);
            trim(value);
            arg.first = key;
            arg.second.clear();
            arg.second.push_back(value);
        } else if (line.back() == ':') {
            params.push_back(arg);
            line.pop_back();
            trim(line);
            arg.first = line;
            arg.second.clear();
            continue;
        } else if (line.front() == '-') {
            line = line.substr(1);
            trim(line);
            arg.second.push_back(line);
            continue;
        }
    }
    params.push_back(arg);
    params.erase(params.begin());

    return params;
}

// Function to generate all possible combinations of parameters
void generateCombinations(const std::vector<std::string> &param_names, const std::vector<std::vector<std::string>> &param_values, bool is_kv) {
    size_t total_combinations = 1;
    for (const auto &values : param_values) {
        total_combinations *= values.size();
    }
    std::vector<std::string> current_params(param_names.size(), "");

    for (int i = 0; i < (int)total_combinations; ++i) {
        int idx = i;
        for (int j = param_values.size() - 1; j >= 0; --j) {
            current_params[j] = param_values[j][idx % param_values[j].size()];
            if (is_kv) {
                current_params[j] = param_names[j] + "=" + current_params[j];
            }
            idx /= param_values[j].size();
        }
        if (config.program_file.find("tester") != std::string::npos) {
            runTester(current_params);
        } else {
            runProgram(current_params);
        }
    }
}

int readArgs(int argc, char *argv[]) {
    if (argc <= 3) {
        std::cout << "tester: too few arguments" << std::endl;
        std::cout << "Usage: tester <PROGRAM> <PARAMETERS_FILE> <RESULTS_FILE> [--kv]" << std::endl;
        std::cout << std::endl;
        std::cout << "Testing PROGRAM by executing all combinations of parameters from PARAMETERS_FILE and saving" << std::endl;
        std::cout << "results in the RESULTS_FILE in .csv format. Program itself must write csv lines to stdin and" << std::endl;
        std::cout << "tester will write it to RESULTS_FILE." << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --kv            ensure that parameters should be passed to program in 'key=value' format" << std::endl;
        std::cout << "                    instead of just 'value'" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  tester local_search params_local_search.yml results.csv --kv" << std::endl;
        std::cout << "  tester genetic params_genetic.yml results.csv" << std::endl;
        std::cout << "  tester tester params_tester.yml results.csv" << std::endl;
        // tester tester params_tester.yml results.csv
        //
        // params_tester.yml:
        // program:
        // - local_search
        // params:
        // - params_local_search1.yml
        // - params_local_search2.yml
        // - params_local_search3.yml
        // - params_local_search4.yml
        // results:
        // - results.csv
        // kv:
        // - --kv
        //
        return 1;
    } else {
        config.program_file = std::string(argv[1]);
        config.params_file = std::string(argv[2]);
        config.results_file = std::string(argv[3]);
        config.is_kv = argc >= 5 && std::string(argv[4]) == "--kv";
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {

    int result = readArgs(argc, argv);
    if (result != 0) {
        return result;
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> params = parseYAML(config.params_file);

    // printParams(params);

    // Extract parameter names and values into separate vectors
    std::vector<std::string> param_names;
    std::vector<std::vector<std::string>> param_values;

    for (const auto &param : params) {
        param_names.push_back(param.first);
        param_values.push_back(param.second);
    }

    results_file.open(config.results_file, std::ios::app);
    if (!results_file) {
        std::cerr << "Error opening results file!" << std::endl;
        return 1;
    }

    generateCombinations(param_names, param_values, config.is_kv);

    results_file.close();

    return 0;
}