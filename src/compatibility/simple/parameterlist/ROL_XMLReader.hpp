// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_XMLREADER_HPP
#define ROL_XMLREADER_HPP

#include <pugixml.hpp>
#include <string>
#include <sstream>
#include <stdexcept>
#include "ROL_ParameterList.hpp"
#include "ROL_Ptr.hpp"

namespace ROL {

class XMLReader {
public:
    static ROL::Ptr<ParameterList> readParametersFromFile(const std::string& filename) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(filename.c_str());
        
        if (!result) {
            throw std::runtime_error("XMLReader: Failed to load XML file: " + filename + 
                                   " Error: " + result.description());
        }
        
        ROL::Ptr<ParameterList> params = ROL::makePtr<ParameterList>();
        
        // Find the root ParameterList element
        pugi::xml_node root = doc.child("ParameterList");
        if (!root) {
            throw std::runtime_error("XMLReader: No ParameterList element found in XML file");
        }
        
        parseParameterList(root, *params);
        return params;
    }

private:
    static void parseParameterList(const pugi::xml_node& node, ParameterList& params) {
        for (pugi::xml_node child : node.children()) {
            std::string name = child.name();
            
            if (name == "Parameter") {
                parseParameter(child, params);
            } else if (name == "ParameterList") {
                parseSublist(child, params);
            }
        }
    }
    
    static void parseParameter(const pugi::xml_node& node, ParameterList& params) {
        std::string name = node.attribute("name").as_string();
        std::string type = node.attribute("type").as_string();
        std::string value = node.attribute("value").as_string();
        
        if (name.empty()) {
            throw std::runtime_error("XMLReader: Parameter element missing 'name' attribute");
        }
        
        if (type.empty()) {
            throw std::runtime_error("XMLReader: Parameter element missing 'type' attribute");
        }
        
        // Parse value based on type
        if (type == "double") {
            params.set(name, std::stod(value));
        } else if (type == "int") {
            params.set(name, std::stoi(value));
        } else if (type == "bool") {
            bool bval = (value == "true" || value == "1");
            params.set(name, bval);
        } else if (type == "string") {
            params.set(name, value);
        } else {
            // Default to string for unknown types
            params.set(name, value);
        }
    }
    
    static void parseSublist(const pugi::xml_node& node, ParameterList& params) {
        std::string name = node.attribute("name").as_string();
        if (name.empty()) {
            throw std::runtime_error("XMLReader: ParameterList element missing 'name' attribute");
        }
        
        ParameterList& sublist = params.sublist(name);
        parseParameterList(node, sublist);
    }
};

// Global function for compatibility
inline ROL::Ptr<ParameterList> getParametersFromXmlFile(const std::string& filename) {
    return XMLReader::readParametersFromFile(filename);
}

} // namespace ROL

#endif // ROL_XMLREADER_HPP
