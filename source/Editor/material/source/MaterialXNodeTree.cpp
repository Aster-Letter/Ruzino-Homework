#include "MCore/MaterialXNodeTree.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
std::shared_ptr<MaterialXNodeTree> createMaterialXNodeTree(
    const std::string& materialFilename)
{
    std::shared_ptr<NodeTreeDescriptor> descriptor =
        std::make_shared<MaterialXNodeTreeDescriptor>();
    return std::make_shared<MaterialXNodeTree>(materialFilename, descriptor);
}

class MaterialXSocketDeclaration : public SocketDeclaration {
   public:
    NodeSocket* build(NodeTree* ntree, Node* node) const override;
};

static mx::NodePtr getMaterialXNode(Node* node)
{
    auto cast = node->storage.try_cast<mx::NodePtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

static mx::NodeGraphPtr getMaterialXNodeGraph(Node* node)
{
    auto cast = node->storage.try_cast<mx::NodeGraphPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

static mx::InputPtr getMaterialXInput(Node* node)
{
    auto cast = node->storage.try_cast<mx::InputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

static mx::OutputPtr getMaterialXOutput(Node* node)
{
    auto cast = node->storage.try_cast<mx::OutputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

static mx::InputPtr getMaterialXPinInput(NodeSocket* socket)
{
    auto cast = socket->dataField.value.try_cast<mx::InputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

static mx::OutputPtr getMaterialXPinOutput(NodeSocket* socket)
{
    auto cast = socket->dataField.value.try_cast<mx::OutputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

NodeSocket* MaterialXSocketDeclaration::build(NodeTree* ntree, Node* node) const
{
    NodeSocket* socket = node->add_socket(
        type.info().name().data(),
        this->identifier.c_str(),
        this->name.c_str(),
        this->in_out);
    update_default_value(socket);

    return socket;
}

void MaterialXNodeTree::loadStandardLibraries()
{
    // Initialize the standard library.
    try {
        _stdLib = mx::createDocument();
        _xincludeFiles =
            mx::loadLibraries(_libraryFolders, _searchPath, _stdLib);
        if (_xincludeFiles.empty()) {
            std::cerr << "Could not find standard data libraries on the given "
                         "search path: "
                      << _searchPath.asString() << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "Failed to load standard data libraries: " << e.what()
                  << std::endl;
        return;
    }
}

mx::DocumentPtr MaterialXNodeTree::loadDocument(const mx::FilePath& filename)
{
    mx::FilePathVec libraryFolders = { "libraries" };
    _libraryFolders = libraryFolders;
    mx::XmlReadOptions readOptions;
    readOptions.readXIncludeFunction = [](mx::DocumentPtr doc,
                                          const mx::FilePath& filename,
                                          const mx::FileSearchPath& searchPath,
                                          const mx::XmlReadOptions* options) {
        mx::FilePath resolvedFilename = searchPath.find(filename);
        if (resolvedFilename.exists()) {
            try {
                readFromXmlFile(doc, resolvedFilename, searchPath, options);
            }
            catch (mx::Exception& e) {
                std::cerr << "Failed to read include file: "
                          << filename.asString() << ". "
                          << std::string(e.what()) << std::endl;
            }
        }
        else {
            std::cerr << "Include file not found: " << filename.asString()
                      << std::endl;
        }
    };

    mx::DocumentPtr doc = mx::createDocument();
    try {
        if (!filename.isEmpty()) {
            mx::readFromXmlFile(doc, filename, _searchPath, &readOptions);
            doc->importLibrary(_stdLib);
            std::string message;
            if (!doc->validate(&message)) {
                std::cerr << "*** Validation warnings for "
                          << filename.asString() << " ***" << std::endl;
                std::cerr << message << std::endl;
            }

            // Cache the currently loaded file
            _materialFilename = filename;
        }
    }
    catch (mx::Exception& e) {
        std::cerr << "Failed to read file: " << filename.asString() << ": \""
                  << std::string(e.what()) << "\"" << std::endl;
    }
    //_graphStack = std::stack<std::vector<UiNodePtr>>();
    //_pinStack = std::stack<std::vector<UiPinPtr>>();
    return doc;
}

void MaterialXNodeTree::buildUiBaseGraph(mx::DocumentPtr doc)
{
    std::vector<mx::NodeGraphPtr> nodeGraphs = doc->getNodeGraphs();
    std::vector<mx::InputPtr> inputNodes = doc->getActiveInputs();
    std::vector<mx::OutputPtr> outputNodes = doc->getOutputs();
    std::vector<mx::NodePtr> docNodes = doc->getNodes();

    mx::ElementPredicate includeElement = getElementPredicate();

    this->clear();
    // nodes.clear();
    // links.clear();
    //_currEdge.clear();
    // sockets.clear();
    //_graphTotalSize = 1;

    // Create UiNodes for nodes that belong to the document so they are not in a
    // nodegraph
    for (mx::NodePtr node : docNodes) {
        if (!includeElement(node))
            continue;
        std::string getType = node->getType();

        auto currNode = add_node(getType.c_str());
        currNode->ui_name = node->getName();

        // auto currNode = std::make_shared<UiNode>(name, _graphTotalSize);
        currNode->storage = node;
        setUiNodeInfo(currNode, node->getType(), node->getCategory());
    }

    // Create UiNodes for the nodegraph
    for (mx::NodeGraphPtr nodeGraph : nodeGraphs) {
        if (!includeElement(nodeGraph))
            continue;
        auto currNode = add_node("groupnode");
        currNode->ui_name = nodeGraph->getName();
        currNode->storage = nodeGraph;
        setUiNodeInfo(currNode, nodeGraph->getType(), nodeGraph->getCategory());
    }
    for (mx::InputPtr input : inputNodes) {
        if (!includeElement(input))
            continue;
        auto currNode = add_node(input->getType().c_str());
        currNode->ui_name = input->getName();
        currNode->storage = input;
        setUiNodeInfo(currNode, input->getType(), input->getCategory());
    }
    for (mx::OutputPtr output : outputNodes) {
        if (!includeElement(output))
            continue;
        auto currNode = add_node(output->getType().c_str());
        currNode->ui_name = output->getName();
        currNode->storage = output;
        setUiNodeInfo(currNode, output->getType(), output->getCategory());
    }

    for (auto& node : nodes) {
        std::cout << node->typeinfo->id_name
                  << ", id name: " << node->typeinfo->id_name << std::endl;
    }

    // Create edges for nodegraphs
    for (mx::NodeGraphPtr graph : nodeGraphs) {
        for (mx::InputPtr input : graph->getActiveInputs()) {
            int downNum = -1;
            int upNum = -1;
            mx::string nodeGraphName = input->getNodeGraphString();
            mx::NodePtr connectedNode = input->getConnectedNode();
            if (!nodeGraphName.empty()) {
                downNum = findNode(graph->getName(), "nodegraph");
                upNum = findNode(nodeGraphName, "nodegraph");
            }
            else if (connectedNode) {
                downNum = findNode(graph->getName(), "nodegraph");
                upNum = findNode(connectedNode->getName(), "node");
            }

            if (downNum == -1 || upNum == -1)
                continue;

            auto& up_node = nodes[upNum];
            auto& down_node = nodes[downNum];

            auto input_socket =
                down_node->get_input_socket(input->getName().c_str());

            auto output =
                getOutputPin(down_node.get(), up_node.get(), input_socket);

            add_link(output, input_socket->ID);
        }
    }

    // Create edges for surface and material nodes
    for (mx::NodePtr node : docNodes) {
        mx::NodeDefPtr nD = node->getNodeDef(node->getName());
        for (mx::InputPtr input : node->getActiveInputs()) {
            mx::string nodeGraphName = input->getNodeGraphString();
            mx::NodePtr connectedNode = input->getConnectedNode();
            mx::OutputPtr connectedOutput = input->getConnectedOutput();
            int upNum = -1;
            int downNum = -1;
            if (!nodeGraphName.empty()) {
                upNum = findNode(nodeGraphName, "nodegraph");
                downNum = findNode(node->getName(), "node");
            }
            else if (connectedNode) {
                upNum = findNode(connectedNode->getName(), "node");
                downNum = findNode(node->getName(), "node");
            }
            else if (connectedOutput) {
                upNum = findNode(connectedOutput->getName(), "output");
                downNum = findNode(node->getName(), "node");
            }
            else if (!input->getInterfaceName().empty()) {
                upNum = findNode(input->getInterfaceName(), "input");
                downNum = findNode(node->getName(), "node");
            }

            if (downNum == -1 || upNum == -1)
                continue;

            auto& up_node = nodes[upNum];
            auto& down_node = nodes[downNum];

            auto input_socket =
                down_node->get_input_socket(input->getName().c_str());

            auto output =
                getOutputPin(down_node.get(), up_node.get(), input_socket);

            add_link(output, input_socket->ID);
        }
    }
}

void MaterialXNodeTree::buildUiNodeGraph(const mx::NodeGraphPtr& nodeGraphs)
{
    //// Clear all values so that ids can start with 0 or 1
    // nodes.clear();
    // links.clear();
    //_currEdge.clear();
    // sockets.clear();
    //_graphTotalSize = 1;
    // if (nodeGraphs) {
    //     mx::NodeGraphPtr nodeGraph = nodeGraphs;
    //     std::vector<mx::ElementPtr> children = nodeGraph->topologicalSort();
    //     mx::NodeDefPtr nodeDef = nodeGraph->getNodeDef();
    //     mx::NodeDefPtr currNodeDef;

    //    // Create input nodes
    //    if (nodeDef) {
    //        std::vector<mx::InputPtr> inputs = nodeDef->getActiveInputs();

    //        for (mx::InputPtr input : inputs) {
    //            auto currNode =
    //                std::make_shared<UiNode>(input->getName(),
    //                _graphTotalSize);
    //            currNode->setInput(input);
    //            setUiNodeInfo(currNode, input->getType(),
    //            input->getCategory());
    //        }
    //    }

    //    // Search node graph children to create uiNodes
    //    for (mx::ElementPtr elem : children) {
    //        mx::NodePtr node = elem->asA<mx::Node>();
    //        mx::InputPtr input = elem->asA<mx::Input>();
    //        mx::OutputPtr output = elem->asA<mx::Output>();
    //        std::string name = elem->getName();
    //        auto currNode = std::make_shared<UiNode>(name, _graphTotalSize);
    //        if (node) {
    //            currNode->setNode(node);
    //            setUiNodeInfo(currNode, node->getType(), node->getCategory());
    //        }
    //        else if (input) {
    //            currNode->setInput(input);
    //            setUiNodeInfo(currNode, input->getType(),
    //            input->getCategory());
    //        }
    //        else if (output) {
    //            currNode->setOutput(output);
    //            setUiNodeInfo(
    //                currNode, output->getType(), output->getCategory());
    //        }
    //    }

    //    // Write out all connections.
    //    std::set<mx::Edge> processedEdges;
    //    for (mx::OutputPtr output : nodeGraph->getOutputs()) {
    //        for (mx::Edge edge : output->traverseGraph()) {
    //            if (!processedEdges.count(edge)) {
    //                mx::ElementPtr upstreamElem = edge.getUpstreamElement();
    //                mx::ElementPtr downstreamElem =
    //                edge.getDownstreamElement(); mx::ElementPtr connectingElem
    //                = edge.getConnectingElement();

    //                mx::NodePtr upstreamNode = upstreamElem->asA<mx::Node>();
    //                mx::NodePtr downstreamNode =
    //                    downstreamElem->asA<mx::Node>();
    //                mx::InputPtr upstreamInput =
    //                upstreamElem->asA<mx::Input>(); mx::InputPtr
    //                downstreamInput =
    //                    downstreamElem->asA<mx::Input>();
    //                mx::OutputPtr upstreamOutput =
    //                    upstreamElem->asA<mx::Output>();
    //                mx::OutputPtr downstreamOutput =
    //                    downstreamElem->asA<mx::Output>();
    //                std::string downName = downstreamElem->getName();
    //                std::string upName = upstreamElem->getName();
    //                std::string upstreamType;
    //                std::string downstreamType;
    //                if (upstreamNode) {
    //                    upstreamType = "node";
    //                }
    //                else if (upstreamInput) {
    //                    upstreamType = "input";
    //                }
    //                else if (upstreamOutput) {
    //                    upstreamType = "output";
    //                }
    //                if (downstreamNode) {
    //                    downstreamType = "node";
    //                }
    //                else if (downstreamInput) {
    //                    downstreamType = "input";
    //                }
    //                else if (downstreamOutput) {
    //                    downstreamType = "output";
    //                }
    //                int upNode = findNode(upName, upstreamType);
    //                int downNode = findNode(downName, downstreamType);
    //                if (downNode > 0 && upNode > 0 &&
    //                    nodes[downNode]->getOutput()) {
    //                    // Create edges for the output nodes
    //                    UiEdge newEdge =
    //                        UiEdge(nodes[upNode], nodes[downNode], nullptr);
    //                    if (!edgeExists(newEdge)) {
    //                        nodes[downNode]->edges.push_back(newEdge);
    //                        nodes[downNode]->setInputNodeNum(1);
    //                        nodes[upNode]->setOutputConnection(nodes[downNode]);
    //                        _currEdge.push_back(newEdge);
    //                    }
    //                }
    //                else if (connectingElem) {
    //                    mx::InputPtr connectingInput =
    //                        connectingElem->asA<mx::Input>();

    //                    if (connectingInput) {
    //                        if ((upNode >= 0) && (downNode >= 0)) {
    //                            UiEdge newEdge = UiEdge(
    //                                nodes[upNode],
    //                                nodes[downNode],
    //                                connectingInput);
    //                            if (!edgeExists(newEdge)) {
    //                                nodes[downNode]->edges.push_back(newEdge);
    //                                nodes[downNode]->setInputNodeNum(1);
    //                                nodes[upNode]->setOutputConnection(
    //                                    nodes[downNode]);
    //                                _currEdge.push_back(newEdge);
    //                            }
    //                        }
    //                    }
    //                }
    //                if (upstreamNode) {
    //                    std::vector<mx::InputPtr> ins =
    //                        upstreamNode->getActiveInputs();
    //                    for (mx::InputPtr input : ins) {
    //                        // Connect input nodes
    //                        if (input->hasInterfaceName()) {
    //                            std::string interfaceName =
    //                                input->getInterfaceName();
    //                            int newUp = findNode(interfaceName, "input");
    //                            if (newUp >= 0) {
    //                                mx::InputPtr inputP =
    //                                    std::make_shared<mx::Input>(
    //                                        downstreamElem, input->getName());
    //                                UiEdge newEdge = UiEdge(
    //                                    nodes[newUp], nodes[upNode], input);
    //                                if (!edgeExists(newEdge)) {
    //                                    nodes[upNode]->edges.push_back(newEdge);
    //                                    nodes[upNode]->setInputNodeNum(1);
    //                                    nodes[newUp]->setOutputConnection(
    //                                        nodes[upNode]);
    //                                    _currEdge.push_back(newEdge);
    //                                }
    //                            }
    //                        }
    //                    }
    //                }

    //                processedEdges.insert(edge);
    //            }
    //        }
    //    }

    //    // Second pass to catch all of the connections that arent part of an
    //    // output
    //    for (mx::ElementPtr elem : children) {
    //        mx::NodePtr node = elem->asA<mx::Node>();
    //        mx::InputPtr inputElem = elem->asA<mx::Input>();
    //        mx::OutputPtr output = elem->asA<mx::Output>();
    //        if (node) {
    //            std::vector<mx::InputPtr> inputs = node->getActiveInputs();
    //            for (mx::InputPtr input : inputs) {
    //                mx::NodePtr upNode = input->getConnectedNode();
    //                if (upNode) {
    //                    int upNum = findNode(upNode->getName(), "node");
    //                    int downNode = findNode(node->getName(), "node");
    //                    if ((upNum >= 0) && (downNode >= 0)) {
    //                        UiEdge newEdge =
    //                            UiEdge(nodes[upNum], nodes[downNode], input);
    //                        if (!edgeExists(newEdge)) {
    //                            nodes[downNode]->edges.push_back(newEdge);
    //                            nodes[downNode]->setInputNodeNum(1);
    //                            nodes[upNum]->setOutputConnection(
    //                                nodes[downNode]);
    //                            _currEdge.push_back(newEdge);
    //                        }
    //                    }
    //                }
    //                else if (input->getInterfaceInput()) {
    //                    int upNum = findNode(
    //                        input->getInterfaceInput()->getName(), "input");
    //                    int downNode = findNode(node->getName(), "node");
    //                    if ((upNum >= 0) && (downNode >= 0)) {
    //                        UiEdge newEdge =
    //                            UiEdge(nodes[upNum], nodes[downNode], input);
    //                        if (!edgeExists(newEdge)) {
    //                            nodes[downNode]->edges.push_back(newEdge);
    //                            nodes[downNode]->setInputNodeNum(1);
    //                            nodes[upNum]->setOutputConnection(
    //                                nodes[downNode]);
    //                            _currEdge.push_back(newEdge);
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //        else if (output) {
    //            mx::NodePtr upNode = output->getConnectedNode();
    //            if (upNode) {
    //                int upNum = findNode(upNode->getName(), "node");
    //                int downNode = findNode(output->getName(), "output");
    //                UiEdge newEdge =
    //                    UiEdge(nodes[upNum], nodes[downNode], nullptr);
    //                if (!edgeExists(newEdge)) {
    //                    nodes[downNode]->edges.push_back(newEdge);
    //                    nodes[downNode]->setInputNodeNum(1);
    //                    nodes[upNum]->setOutputConnection(nodes[downNode]);
    //                    _currEdge.push_back(newEdge);
    //                }
    //            }
    //        }
    //    }
    //}
}
void MaterialXNodeTree::setUiNodeInfo(
    UiNodePtr node,
    const std::string& type,
    const std::string& category)
{
    // Do nothing if sockets are already set up.
    if (!node->typeinfo->static_declaration.items.empty())
        return;

    // Lambda to create and add a socket declaration.
    auto createSocketDeclaration =
        [&](auto socketElement, PinKind kind, auto& targetVector) {
            auto socketDecl = std::make_shared<MaterialXSocketDeclaration>();
            std::string socketName = socketElement->getName();
            std::string socketCategory = socketElement->getCategory();

            socketDecl->name = socketName;
            socketDecl->identifier = socketName;
            // Setup type using socket category.
            entt::meta<void>(get_entt_ctx())
                .type(entt::hashed_string(socketCategory.c_str()));
            socketDecl->type =
                get_socket_type(entt::hashed_string(socketCategory.c_str()));
            socketDecl->in_out = kind;

            node->typeinfo->static_declaration.items.push_back(socketDecl);
            targetVector.push_back(socketDecl.get());
        };

    // If the node stores a NodeGraph.
    if (node->storage.allow_cast<mx::NodeGraphPtr>()) {
        auto nodeGraph = node->storage.cast<mx::NodeGraphPtr>();

        for (mx::OutputPtr out : nodeGraph->getOutputs())
            createSocketDeclaration(
                out,
                PinKind::Output,
                node->typeinfo->static_declaration.outputs);
        for (mx::InputPtr in : nodeGraph->getInputs())
            createSocketDeclaration(
                in, PinKind::Input, node->typeinfo->static_declaration.inputs);
    }
    // If the node stores a regular Node.
    else if (node->storage.allow_cast<mx::NodePtr>()) {
        auto mnode = node->storage.cast<mx::NodePtr>();
        mx::NodeDefPtr nodeDef = mnode->getNodeDef(mnode->getName());
        if (nodeDef) {
            for (mx::InputPtr in : nodeDef->getActiveInputs())
                createSocketDeclaration(
                    in,
                    PinKind::Input,
                    node->typeinfo->static_declaration.inputs);
            for (mx::OutputPtr out : nodeDef->getActiveOutputs())
                createSocketDeclaration(
                    out,
                    PinKind::Output,
                    node->typeinfo->static_declaration.outputs);
        }
    }

    node->refresh_node();

    // If the node stores a NodeGraph.
    if (node->storage.allow_cast<mx::NodeGraphPtr>()) {
        auto nodeGraph = node->storage.cast<mx::NodeGraphPtr>();

        for (int i = 0; i < nodeGraph->getOutputs().size(); i++) {
            auto out = nodeGraph->getOutputs()[i];
            auto socket = node->get_outputs()[i];
            socket->storage = out;
        }

        for (int i = 0; i < nodeGraph->getInputs().size(); i++) {
            auto in = nodeGraph->getInputs()[i];
            auto socket = node->get_input_socket(in->getName().c_str());
            if (socket)
                socket->storage = in;
        }
    }
    // If the node stores a regular Node.
    else if (node->storage.allow_cast<mx::NodePtr>()) {
        auto mnode = node->storage.cast<mx::NodePtr>();
        mx::NodeDefPtr nodeDef = mnode->getNodeDef(mnode->getName());
        if (nodeDef) {
            // Store inputs
            for (auto& in : nodeDef->getActiveInputs()) {
                auto socket = node->get_input_socket(in->getName().c_str());
                if (socket) {
                    // For node inputs, store the actual input from the node,
                    // not the nodedef
                    auto nodeInput = mnode->getInput(in->getName());
                    if (nodeInput)
                        socket->storage = nodeInput;
                    else
                        socket->storage = in;
                }
            }

            // Store outputs
            for (int i = 0; i < nodeDef->getActiveOutputs().size(); i++) {
                auto out = nodeDef->getActiveOutputs()[i];
                auto nodeOutput = mnode->getOutput(out->getName());
                auto socket = node->get_outputs()[i];
                if (socket) {
                    if (nodeOutput)
                        socket->storage = nodeOutput;
                    else
                        socket->storage = out;
                }
            }
        }
    }
}

int MaterialXNodeTree::findNode(int nodeId)
{
    // int count = 0;
    // for (size_t i = 0; i < nodes.size(); i++) {
    //     if (nodes[i]->getId() == nodeId) {
    //         return count;
    //     }
    //     count++;
    // }
    return -1;
}

int MaterialXNodeTree::findNode(
    const std::string& name,
    const std::string& type)
{
    int count = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->getName() == name) {
            if (type == "node" && getMaterialXNode(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "input" &&
                getMaterialXInput(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "output" &&
                getMaterialXOutput(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "nodegraph" &&
                getMaterialXNodeGraph(nodes[i].get()) != nullptr) {
                return count;
            }
        }
        count++;
    }
    return -1;
}

void MaterialXNodeTree::addNode(
    const std::string& category,
    const std::string& name,
    const std::string& type)
{
    // mx::NodePtr node = nullptr;
    // std::vector<mx::NodeDefPtr> matchingNodeDefs;

    //// Create document or node graph is there is not already one
    // if (category == "output") {
    //     std::string outName = "";
    //     mx::OutputPtr newOut;
    //     // add output as child of correct parent and create valid name
    //     outName = _currGraphElem->createValidChildName(name);
    //     newOut = _currGraphElem->addOutput(outName, type);
    //     auto outputNode =
    //         std::make_shared<UiNode>(outName, int(++_graphTotalSize));
    //     outputNode->setOutput(newOut);
    //     setUiNodeInfo(outputNode, type, category);
    //     return;
    // }
    // if (category == "input") {
    //     std::string inName = "";
    //     mx::InputPtr newIn = nullptr;

    //    // Add input as child of correct parent and create valid name
    //    inName = _currGraphElem->createValidChildName(name);
    //    newIn = _currGraphElem->addInput(inName, type);
    //    auto inputNode =
    //        std::make_shared<UiNode>(inName, int(++_graphTotalSize));
    //    setDefaults(newIn);
    //    inputNode->setInput(newIn);
    //    setUiNodeInfo(inputNode, type, category);
    //    return;
    //}
    // else if (category == "group") {
    //    auto groupNode = std::make_shared<UiNode>(name,
    //    int(++_graphTotalSize));

    //    // Set message of group UiNode in order to identify it as such
    //    groupNode->setMessage("Comment");
    //    setUiNodeInfo(groupNode, type, "group");

    //    // Create ui portions of group node
    //    buildGroupNode(nodes.back());
    //    return;
    //}
    // else if (category == "nodegraph") {
    //    // Create new mx::NodeGraph and set as current node graph
    //    _graphDoc->addNodeGraph();
    //    std::string nodeGraphName =
    //        _graphDoc->getNodeGraphs().back()->getName();
    //    auto nodeGraphNode =
    //        std::make_shared<UiNode>(nodeGraphName,
    //        int(++_graphTotalSize));

    //    // Set mx::Nodegraph as node graph for uiNode
    //    nodeGraphNode->setNodeGraph(_graphDoc->getNodeGraphs().back());

    //    setUiNodeInfo(nodeGraphNode, type, "nodegraph");
    //    return;
    //}
    // else {
    //    matchingNodeDefs = _graphDoc->getMatchingNodeDefs(category);
    //    for (mx::NodeDefPtr nodedef : matchingNodeDefs) {
    //        std::string userNodeDefName =
    //            getUserNodeDefName(nodedef->getName());
    //        if (userNodeDefName == name) {
    //            node = _currGraphElem->addNodeInstance(
    //                nodedef, _currGraphElem->createValidChildName(name));
    //        }
    //    }
    //}

    // if (node) {
    //     int num = 0;
    //     int countDef = 0;
    //     for (size_t i = 0; i < matchingNodeDefs.size(); i++) {
    //         std::string userNodeDefName =
    //             getUserNodeDefName(matchingNodeDefs[i]->getName());
    //         if (userNodeDefName == name) {
    //             num = countDef;
    //         }
    //         countDef++;
    //     }
    //     std::vector<mx::InputPtr> defInputs =
    //         matchingNodeDefs[num]->getActiveInputs();

    //    // Add inputs to UiNode as pins so that we can later add them to
    //    the
    //    // node if necessary
    //    auto newNode =
    //        std::make_shared<UiNode>(node->getName(),
    //        int(++_graphTotalSize));
    //    newNode->setCategory(category);
    //    newNode->setType(type);
    //    newNode->setNode(node);
    //    newNode->_showAllInputs = true;
    //    node->setType(type);
    //    ++_graphTotalSize;
    //    for (mx::InputPtr input : defInputs) {
    //        UiPinPtr inPin = std::make_shared<UiPin>(
    //            _graphTotalSize,
    //            &*input->getName().begin(),
    //            input->getType(),
    //            newNode,
    //            ax::NodeEditor::PinKind::Input,
    //            input,
    //            nullptr);
    //        newNode->inputPins.push_back(inPin);
    //        sockets.push_back(inPin);
    //        ++_graphTotalSize;
    //    }
    //    std::vector<mx::OutputPtr> defOutputs =
    //        matchingNodeDefs[num]->getActiveOutputs();
    //    for (mx::OutputPtr output : defOutputs) {
    //        UiPinPtr outPin = std::make_shared<UiPin>(
    //            _graphTotalSize,
    //            &*output->getName().begin(),
    //            output->getType(),
    //            newNode,
    //            ax::NodeEditor::PinKind::Output,
    //            nullptr,
    //            nullptr);
    //        newNode->get_outputs().push_back(outPin);
    //        sockets.push_back(outPin);
    //        ++_graphTotalSize;
    //    }

    //    nodes.push_back(std::move(newNode));
    //    updateMaterials();
    //}
}

SocketID MaterialXNodeTree::getOutputPin(
    UiNodePtr node,
    UiNodePtr upNode,
    UiPinPtr input)
{
    if (getMaterialXNodeGraph(upNode) != nullptr) {
        // For nodegraph need to get the correct output pin according to the
        // names of the output nodes
        mx::OutputPtr output;
        if (getMaterialXNode(input->node)) {
            output = getMaterialXNode(input->node)
                         ->getConnectedOutput(input->identifier);
        }
        else if (getMaterialXNodeGraph(input->node)) {
            output = getMaterialXNodeGraph(input->node)
                         ->getConnectedOutput(input->identifier);
        }

        if (output) {
            std::string outName = output->getName();
            for (UiPinPtr outputs : upNode->get_outputs()) {
                if (outputs->identifier == outName) {
                    return outputs->ID;
                }
            }
        }
        return SocketID();
    }
    else {
        // For node need to get the correct output pin based on the output
        // attribute

        // throw std::runtime_error("Not implemented");
        if (!upNode->get_outputs().empty()) {
            std::string outputName = mx::EMPTY_STRING;
            if (getMaterialXPinInput(input)) {
                outputName = getMaterialXPinInput(input)->getOutputString();
            }
            else if (getMaterialXPinOutput(input)) {
                outputName = getMaterialXPinOutput(input)->getOutputString();
            }

            size_t pinIndex = 0;
            if (!outputName.empty()) {
                for (size_t i = 0; i < upNode->get_outputs().size(); i++) {
                    if (upNode->get_outputs()[i]->identifier == outputName) {
                        pinIndex = i;
                        break;
                    }
                }
            }
            return (upNode->get_outputs()[pinIndex]->ID);
        }
        return SocketID();
    }
}

void MaterialXNodeTree::addLink(SocketID startPinId, SocketID endPinId)
{
    //// Prefer to assume left to right - start is an output, end is an
    /// input; / swap if inaccurate
    // if (UiPinPtr inputPin = getPin(endPinId);
    //     inputPin && inputPin->_kind != ed::PinKind::Input) {
    //     auto tmp = startPinId;
    //     startPinId = endPinId;
    //     endPinId = tmp;
    // }

    // int end_attr = int(endPinId.Get());
    // int start_attr = int(startPinId.Get());
    // SocketID outputPinId = startPinId;
    // SocketID inputPinId = endPinId;
    // UiPinPtr outputPin = getPin(outputPinId);
    // UiPinPtr inputPin = getPin(inputPinId);

    // if (!inputPin || !outputPin) {
    //     ed::RejectNewItem();
    //     return;
    // }

    //// Perform type check
    // bool typesMatch = (outputPin->_type == inputPin->_type);
    // if (!typesMatch) {
    //     ed::RejectNewItem();
    //     showLabel(
    //         "Invalid connection due to mismatched types",
    //         ImColor(50, 50, 50, 255));
    //     return;
    // }

    //// Perform kind check
    // bool kindsMatch = (outputPin->_kind == inputPin->_kind);
    // if (kindsMatch) {
    //     ed::RejectNewItem();
    //     showLabel(
    //         "Invalid connection due to same input/output kind",
    //         ImColor(50, 50, 50, 255));
    //     return;
    // }

    // int upNode = getNodeId(outputPinId);
    // int downNode = getNodeId(inputPinId);
    // UiNodePtr uiDownNode = nodes[downNode];
    // UiNodePtr uiUpNode = nodes[upNode];
    // if (!uiDownNode || !uiUpNode) {
    //     ed::RejectNewItem();
    //     return;
    // }

    //// Make sure there is an implementation for node
    // const mx::ShaderGenerator& shadergen =
    //     _renderer->getGenContext().getShaderGenerator();

    //// Prevent direct connecting from input to output
    // if (uiDownNode->getInput() && uiUpNode->getOutput()) {
    //     ed::RejectNewItem();
    //     showLabel(
    //         "Direct connections between inputs and outputs is invalid",
    //         ImColor(50, 50, 50, 255));
    //     return;
    // }

    //// Find the implementation for this nodedef if not an input or output
    //// uinode
    // if (uiDownNode->getInput() && _isNodeGraph) {
    //     ed::RejectNewItem();
    //     showLabel(
    //         "Cannot connect to inputs inside of graph",
    //         ImColor(50, 50, 50, 255));
    //     return;
    // }
    // else if (getMaterialXNode(uiUpNode)) {
    //     mx::ShaderNodeImplPtr impl = shadergen.getImplementation(
    //         *nodes[upNode]->getNode()->getNodeDef(),
    //         _renderer->getGenContext());
    //     if (!impl) {
    //         ed::RejectNewItem();
    //         showLabel(
    //             "Invalid Connection: Node does not have an
    //             implementation", ImColor(50, 50, 50, 255));
    //         return;
    //     }
    // }

    // if (ed::AcceptNewItem()) {
    //     // If the accepting node already has a link, remove it
    //     if (inputPin->_connected) {
    //         for (auto iter = links.begin(); iter != links.end(); ++iter)
    //         {
    //             if (iter->_endAttr == end_attr) {
    //                 // Found existing link - remove it; adapted from
    //                 // deleteLink note: ed::BreakLinks doesn't work as
    //                 the
    //                 // order ends up inaccurate
    //                 deleteLinkInfo(iter->_startAttr, iter->_endAttr);
    //                 links.erase(iter);
    //                 break;
    //             }
    //         }
    //     }

    //    // Since we accepted new link, lets add one to our list of links.
    //    Link link;
    //    link._startAttr = start_attr;
    //    link._endAttr = end_attr;
    //    links.push_back(link);
    //    _frameCount = ImGui::GetFrameCount();
    //    _renderer->setMaterialCompilation(true);

    //    inputPin->addConnection(outputPin);
    //    outputPin->addConnection(inputPin);
    //    outputPin->setConnected(true);
    //    inputPin->setConnected(true);

    //    if (getMaterialXNode(uiDownNode) || getMaterialXNodeGraph(uiDownNode))
    //    {
    //        mx::InputPtr connectingInput = nullptr;
    //        for (UiPinPtr pin : uiDownNode->inputPins) {
    //            if (pin->ID == inputPinId) {
    //                addNodeInput(uiDownNode, getMaterialXPinInput(pin));

    //                // Update value to be empty
    //                if (getMaterialXNode(uiDownNode) &&
    //                    getMaterialXNode(uiDownNode)->getType() ==
    //                        mx::SURFACE_SHADER_TYPE_STRING) {
    //                    if (uiUpNode->getOutput() != nullptr) {
    //                        getMaterialXPinInput(pin)->setConnectedOutput(
    //                            uiUpNode->getOutput());
    //                    }
    //                    else if (uiUpNode->getInput() != nullptr) {
    //                        getMaterialXPinInput(pin)->setConnectedInterfaceName(
    //                            uiUpNode->getName());
    //                    }
    //                    else {
    //                        if (getMaterialXNodeGraph(uiUpNode) != nullptr) {
    //                            for (UiPinPtr outPin :
    //                            uiUpNode->get_outputs()) {
    //                                // Set pin connection to correct
    //                                output if (outPin->ID ==
    //                                outputPinId) {
    //                                    mx::OutputPtr outputs =
    //                                        getMaterialXNodeGraph(uiUpNode)->getOutput(
    //                                            outPin->identifier);
    //                                    getMaterialXPinInput(pin)->setConnectedOutput(
    //                                        outputs);
    //                                }
    //                            }
    //                        }
    //                        else {
    //                            getMaterialXPinInput(pin)->setConnectedNode(
    //                                getMaterialXNode(uiUpNode));
    //                        }
    //                    }
    //                }
    //                else {
    //                    if (uiUpNode->getInput()) {
    //                        getMaterialXPinInput(pin)->setConnectedInterfaceName(
    //                            uiUpNode->getName());
    //                    }
    //                    else {
    //                        if (getMaterialXNode(uiUpNode)) {
    //                            mx::NodePtr upstreamNode =
    //                                nodes[upNode]->getNode();
    //                            mx::NodeDefPtr upstreamNodeDef =
    //                                upstreamNode->getNodeDef();
    //                            bool isMultiOutput =
    //                                upstreamNodeDef
    //                                    ?
    //                                    upstreamNodeDef->getOutputs().size()
    //                                    >
    //                                          1
    //                                    : false;
    //                            if (!isMultiOutput) {
    //                                getMaterialXPinInput(pin)->setConnectedNode(
    //                                    getMaterialXNode(uiUpNode));
    //                            }
    //                            else {
    //                                for (UiPinPtr outPin :
    //                                     nodes[upNode]->get_outputs()) {
    //                                    // Set pin connection to correct
    //                                    // output
    //                                    if (outPin->ID == outputPinId)
    //                                    {
    //                                        mx::OutputPtr outputs =
    //                                            getMaterialXNode(uiUpNode)->getOutput(
    //                                                outPin->identifier);
    //                                        if (!outputs) {
    //                                            outputs =
    //                                                getMaterialXNode(uiUpNode)
    //                                                    ->addOutput(
    //                                                        outPin->identifier,
    //                                                        getMaterialXPinInput(pin)
    //                                                            ->getType());
    //                                        }
    //                                        getMaterialXPinInput(pin)->setConnectedOutput(
    //                                            outputs);
    //                                    }
    //                                }
    //                            }
    //                        }
    //                        else if (getMaterialXNodeGraph(uiUpNode)) {
    //                            for (UiPinPtr outPin :
    //                            uiUpNode->get_outputs()) {
    //                                // Set pin connection to correct
    //                                output if (outPin->ID ==
    //                                outputPinId) {
    //                                    mx::OutputPtr outputs =
    //                                        getMaterialXNodeGraph(uiUpNode)->getOutput(
    //                                            outPin->identifier);
    //                                    getMaterialXPinInput(pin)->setConnectedOutput(
    //                                        outputs);
    //                                }
    //                            }
    //                        }
    //                    }
    //                }

    //                pin->setConnected(true);
    //                connectingInput = getMaterialXPinInput(pin);
    //                break;
    //            }
    //        }

    //        // Create new edge and set edge information
    //        createEdge(nodes[upNode], nodes[downNode], connectingInput);
    //    }
    //    else if (nodes[downNode]->getOutput() != nullptr) {
    //        mx::InputPtr connectingInput = nullptr;
    //        nodes[downNode]->getOutput()->setConnectedNode(
    //            nodes[upNode]->getNode());

    //        // Create new edge and set edge information
    //        createEdge(nodes[upNode], nodes[downNode], connectingInput);
    //    }
    //    else {
    //        // Create new edge and set edge info
    //        UiEdge newEdge = UiEdge(nodes[upNode], nodes[downNode],
    //        nullptr); if (!edgeExists(newEdge)) {
    //            nodes[downNode]->edges.push_back(newEdge);
    //            _currEdge.push_back(newEdge);

    //            // Update input node num and output connections
    //            nodes[downNode]->setInputNodeNum(1);
    //            nodes[upNode]->setOutputConnection(nodes[downNode]);
    //        }
    //    }
    //}
}

void MaterialXNodeTree::removeEdge(int downNode, int upNode, UiPinPtr pin)
{
    // int num = nodes[downNode]->getEdgeIndex(nodes[upNode]->getId(), pin);
    // if (num != -1) {
    //     if (nodes[downNode]->edges.size() == 1) {
    //         nodes[downNode]->edges.erase(nodes[downNode]->edges.begin() +
    //         0);
    //     }
    //     else if (nodes[downNode]->edges.size() > 1) {
    //         nodes[downNode]->edges.erase(nodes[downNode]->edges.begin() +
    //         num);
    //     }
    // }

    // nodes[downNode]->setInputNodeNum(-1);
    // nodes[upNode]->removeOutputConnection(nodes[downNode]->getName());
}

void MaterialXNodeTree::deleteLink(LinkId deletedLinkId)
{
    //// If you agree that link can be deleted, accept deletion.
    // if (ed::AcceptDeletedItem()) {
    //     _renderer->setMaterialCompilation(true);
    //     _frameCount = ImGui::GetFrameCount();
    //     int link_id = int(deletedLinkId.Get());

    //    // Then remove link from your data.
    //    int pos = findLinkPosition(link_id);

    //    // Link start -1 equals node num
    //    Link currLink = links[pos];
    //    deleteLinkInfo(currLink._startAttr, currLink._endAttr);
    //    links.erase(links.begin() + pos);
    //}
}

void MaterialXNodeTree::deleteNode(UiNodePtr node)
{
    //    // Delete link
    //    for (UiPinPtr inputPin : node->inputPins) {
    //        UiNodePtr upNode = node->getConnectedNode(inputPin->identifier);
    //        if (upNode) {
    //            upNode->removeOutputConnection(node->getName());
    //            int num = node->getEdgeIndex(upNode->getId(), inputPin);
    //
    //            // Erase edge between node and up node
    //            if (num != -1) {
    //                if (node->edges.size() == 1) {
    //                    node->edges.erase(node->edges.begin() + 0);
    //                }
    //                else if (node->edges.size() > 1) {
    //                    node->edges.erase(node->edges.begin() + num);
    //                }
    //            }
    //        }
    //    }
    //
    //    for (UiPinPtr outputPin : node->get_outputs()) {
    //        // Update downNode info
    //        for (UiPinPtr pin : outputPin.get()->getConnections()) {
    //            mx::ValuePtr val;
    //            if (pin->getMaterialXNode(node)) {
    //                mx::NodeDefPtr nodeDef =
    //                pin->getMaterialXNode(node)->getNodeDef(
    //                    pin->getMaterialXNode(node)->getName());
    //                val =
    //                    nodeDef->getActiveInput(getMaterialXPinInput(pin)->getName())->getValue();
    //                if (pin->getMaterialXNode(node)->getType() ==
    //                    mx::SURFACE_SHADER_TYPE_STRING) {
    //                    getMaterialXPinInput(pin)->setConnectedOutput(nullptr);
    //                }
    //                else {
    //                    getMaterialXPinInput(pin)->setConnectedNode(nullptr);
    //                }
    //                if (node->getInput()) {
    //                    // Remove interface in order to set the default of
    //                    the
    //                    // input
    //                    getMaterialXPinInput(pin)->setConnectedInterfaceName(mx::EMPTY_STRING);
    //                    setDefaults(getMaterialXPinInput(pin));
    //                    setDefaults(node->getInput());
    //                }
    //            }
    //            else if (pin->getMaterialXNodeGraph(node)) {
    //                if (node->getInput()) {
    //                    pin->getMaterialXNodeGraph(node)
    //                        ->getInput(pin->identifier)
    //                        ->setConnectedInterfaceName(mx::EMPTY_STRING);
    //                    setDefaults(node->getInput());
    //                }
    //                getMaterialXPinInput(pin)->setConnectedNode(nullptr);
    //                pin->setConnected(false);
    //                setDefaults(getMaterialXPinInput(pin));
    //            }
    //
    //            pin->setConnected(false);
    //            if (val) {
    //                getMaterialXPinInput(pin)->setValueString(val->getValueString());
    //            }
    //
    //            int num = pin->node->getEdgeIndex(node->getId(), pin);
    //            if (num != -1) {
    //                if (pin->node->edges.size() == 1) {
    //                    pin->node->edges.erase(
    //                        pin->node->edges.begin() + 0);
    //                }
    //                else if (pin->node->edges.size() > 1) {
    //                    pin->node->edges.erase(
    //                        pin->node->edges.begin() + num);
    //                }
    //            }
    //
    //            pin->node->setInputNodeNum(-1);
    //
    //            // Not really necessary since it will be deleted
    //            node->removeOutputConnection(pin->node->getName());
    //        }
    //    }
    //
    //    // Remove from NodeGraph
    //    // All link information is handled in delete link which is called
    //    before
    //    // this
    //    int nodeNum = findNode(node->getId());
    //    _currGraphElem->removeChild(node->getName());
    //    nodes.erase(nodes.begin() + nodeNum);
}

bool MaterialXNodeTree::edgeExists(UiEdge newEdge)
{
    // if (_currEdge.size() > 0) {
    //     for (UiEdge edge : _currEdge) {
    //         if (edge.getDown()->getId() == newEdge.getDown()->getId()) {
    //             if (edge.getUp()->getId() == newEdge.getUp()->getId()) {
    //                 if (edge.getInput() == newEdge.getInput()) {
    //                     return true;
    //                 }
    //             }
    //         }
    //         else if (edge.getUp()->getId() == newEdge.getDown()->getId())
    //         {
    //             if (edge.getDown()->getId() == newEdge.getUp()->getId())
    //             {
    //                 if (edge.getInput() == newEdge.getInput()) {
    //                     return true;
    //                 }
    //             }
    //         }
    //     }
    // }
    // else {
    //     return false;
    // }
    return false;
}

void MaterialXNodeTree::addNodeGraphPins()
{
    // for (UiNodePtr node : nodes) {
    //     if (getMaterialXNodeGraph(node)) {
    //         if (node->inputPins.size() !=
    //             getMaterialXNodeGraph(node)->getInputs().size()) {
    //             for (mx::InputPtr input :
    //             getMaterialXNodeGraph(node)->getInputs())
    //             {
    //                 std::string name = input->getName();
    //                 auto result = std::find_if(
    //                     node->inputPins.begin(),
    //                     node->inputPins.end(),
    //                     [name](UiPinPtr x) { return x->identifier == name;
    //                     });
    //                 if (result == node->inputPins.end()) {
    //                     UiPinPtr inPin = std::make_shared<UiPin>(
    //                         ++_graphTotalSize,
    //                         &*input->getName().begin(),
    //                         input->getType(),
    //                         node,
    //                         ax::NodeEditor::PinKind::Input,
    //                         input,
    //                         nullptr);
    //                     node->inputPins.push_back(inPin);
    //                     sockets.push_back(inPin);
    //                     ++_graphTotalSize;
    //                 }
    //             }
    //         }
    //         if (node->get_outputs().size() !=
    //             getMaterialXNodeGraph(node)->getOutputs().size()) {
    //             for (mx::OutputPtr output :
    //                  getMaterialXNodeGraph(node)->getOutputs()) {
    //                 std::string name = output->getName();
    //                 auto result = std::find_if(
    //                     node->get_outputs().begin(),
    //                     node->get_outputs().end(),
    //                     [name](UiPinPtr x) { return x->identifier == name;
    //                     });
    //                 if (result == node->get_outputs().end()) {
    //                     UiPinPtr outPin = std::make_shared<UiPin>(
    //                         ++_graphTotalSize,
    //                         &*output->getName().begin(),
    //                         output->getType(),
    //                         node,
    //                         ax::NodeEditor::PinKind::Output,
    //                         nullptr,
    //                         nullptr);
    //                     ++_graphTotalSize;
    //                     node->get_outputs().push_back(outPin);
    //                     sockets.push_back(outPin);
    //                 }
    //             }
    //         }
    //     }
    // }
}

mx::ElementPredicate MaterialXNodeTree::getElementPredicate() const
{
    return [this](mx::ConstElementPtr elem) {
        if (elem->hasSourceUri()) {
            return (_xincludeFiles.count(elem->getSourceUri()) == 0);
        }
        return true;
    };
}

MaterialXNodeTree::~MaterialXNodeTree()
{
}

Node* MaterialXNodeTree::find_node(NodeId id) const
{
    return NodeTree::find_node(id);
}

Node* MaterialXNodeTree::find_node(const char* identifier) const
{
    return NodeTree::find_node(identifier);
}

Node* MaterialXNodeTree::add_node(const char* str)
{
    return NodeTree::add_node(str);
}

void MaterialXNodeTree::delete_node(Node* nodeId, bool allow_repeat_delete)
{
    NodeTree::delete_node(nodeId, allow_repeat_delete);
}

void MaterialXNodeTree::delete_node(NodeId nodeId, bool allow_repeat_delete)
{
    NodeTree::delete_node(nodeId, allow_repeat_delete);
}

void MaterialXNodeTree::delete_link(
    LinkId linkId,
    bool refresh_topology,
    bool remove_from_group)
{
    NodeTree::delete_link(linkId, refresh_topology, remove_from_group);
}

void MaterialXNodeTree::delete_link(
    NodeLink* link,
    bool refresh_topology,
    bool remove_from_group)
{
    NodeTree::delete_link(link, refresh_topology, remove_from_group);
}

mx::DocumentPtr MaterialXNodeTree::get_mtlx_stdlib()
{
    return _stdLib;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE