#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>



#include <GL/glew.h>    // Include GLEW - OpenGL Extension Wrangler
#include <GLFW/glfw3.h> // GLFW provides a cross-platform interface for creating a graphical context,
// initializing OpenGL and binding inputs

#include <glm/glm.hpp>  // GLM is an optimized math library with syntax to similar to OpenGL Shading Language
#include <glm/gtc/matrix_transform.hpp> // include this to create transformation matrices
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include "camera.h"
#include "shader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Key callback function
void key_callback(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
// Screen parameters
GLuint screenWidth = 1024; // Screen width in pixels
GLuint screenHeight = 768; // Screen height in pixels

GLuint loadTexture(const char* filename);

// Matrix for view and projection
glm::mat4 viewMat; // View matrix
glm::mat4 projMat; // Projection matrix

// RGB colors
glm::vec3 redColor(1.f, 0.f, 0.f); // Red color (RGB: 1,0,0)
glm::vec3 yellowColor(1.f, 1.f, 0.f); // Yellow color (RGB: 1,1,0)
glm::vec3 greenColor(0.f, 1.f, 0.f); // Green color (RGB: 0,1,0)
glm::vec3 blueColor(0.f, 0.f, 1.f); // Blue color (RGB: 0,0,1)

// Grid parameters
GLuint vaoGrid; // Vertex Array Object for grid
GLuint vboGrid; // Vertex Buffer Object for grid
int gridSize; // Size of the grid

// Axes parameters
GLuint vaoAxes; // Vertex Array Object for axes
GLuint vboAxes; // Vertex Buffer Object for axes

// Cube parameters
GLuint vaoCube; // Vertex Array Object for cube
GLuint vboCube; // Vertex Buffer Object for cube
GLuint eboCube; // Element Buffer Object for cube

GLuint vaoCube2; // Vertex Array Object for cube
GLuint vboCube2; // Vertex Buffer Object for cube
GLuint eboCube2; // Element Buffer Object for cube

GLuint vaoSkybox; // Vertex Array Object for cube
GLuint vboSkybox; // Vertex Buffer Object for cube

// Net parameters
GLuint vaoNet; // Vertex Array Object for net
GLuint vboNet; // Vertex Buffer Object for net
int netSize; // Size of the net

GLuint vaoSphere; // Vertex Array Object for net
GLuint vboSphere; // Vertex Buffer Object for net

GLuint vaoBigNet; // Vertex Array Object for net
GLuint vboBigNet; // Vertex Buffer Object for net
int netBigSize; // Size of the net

GLuint shaderProgram; // Shader program ID

// Control parameters
float scale_value = 1.f; // Scale factor

glm::vec3 position_model(0.0f, 0.5f, -30.0f); // Position of the model
glm::vec3 rotation_model(0.0f, -180.0f, 0.0f); // Rotation of the model

glm::vec3 position_modelP2(0.0f, 0.5f, 30.0f);
glm::vec3 rotation_modelP2(0.f); // Rotation of the model

glm::vec3 position_modelP2v2(0.0f, 0.0f, 30.0f);

glm::vec3 rotation_modelv2(-30.0f, 0.0f, 0.0f); // Rotation of the model

glm::vec3 rotation_modelv3(0.0f, 0.0f, 0.0f);


glm::vec3 rotation_modelP2v2(-30.0f, 0.0f, 0.0f); // Rotation of the model

glm::vec3 rotation_modelP2v3(0.0f, 0.0f, 0.0f);

glm::vec3 position_world(0.f); // Position of the world
glm::vec3 rotation_world(0.f); // Rotation of the world

glm::vec3 position_net(0.0f);
glm::vec3 rotation_net(0.f); // Rotation of the model

std::vector<glm::vec3*> positionArray;
std::vector<glm::vec3*> rotationArray;
// Rotation angle for each model
float rotationAngle[4] = { 0.f, 0.f, 0.f, 0.f };

int player = 0;
int renderMode = 2; // Render mode (0 for point, 1 for line, 2 for fill)


// Camera parameters
glm::vec3 cameraPos = glm::vec3(0.0f, 40.0f, 50.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -0.5f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
int cameraSwitch = 0;
float cameraSensitivity = 0.1f;

// camera
Camera camera(cameraPos, cameraUp);




// Mouse input variables
float lastX = (float)screenWidth / 2.0;
float lastY = (float)screenHeight / 2.0;
bool firstMouse = true;

double lastFrame = 0.0;
float deltaTime = 0.0f;
//key press boolean
bool mPressed = false;
bool onePressed = false;
bool twoPressed = false;
bool threePressed = false;
bool fourPressed = false;
bool fivePressed = false;
bool sixPressed = false;
bool spacePressed = false;
bool lPressed = false;
bool pPressed = false;
bool hPressed = false;
bool xPressed = false;

bool spacePositionUpdated = false;

bool textureSwitch = true;
const int debounceDelay = 200;

std::vector<glm::vec3> sphereVertices;

GLuint groundTexture;
GLuint metalTexture;
GLuint tattoo;
GLuint skyTexture;
GLuint ballTexture;
GLuint baseTexture;
GLuint techTexture;
GLuint rockTexture;
GLuint blackTexture;
GLuint woodTexture;
unsigned int sphereVAO = 0;
unsigned int indexCount;

void renderSphere()
{
	if (sphereVAO == 0)
	{
		glGenVertexArrays(1, &sphereVAO);

		unsigned int vbo, ebo;
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		std::vector<glm::vec3> positions;
		std::vector<glm::vec2> uv;
		std::vector<glm::vec3> normals;
		std::vector<unsigned int> indices;

		const unsigned int X_SEGMENTS = 64;
		const unsigned int Y_SEGMENTS = 64;
		const float PI = 3.14159265359f;
		for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
		{
			for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
			{
				float xSegment = (float)x / (float)X_SEGMENTS;
				float ySegment = (float)y / (float)Y_SEGMENTS;
				float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
				float yPos = std::cos(ySegment * PI);
				float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

				positions.push_back(glm::vec3(xPos, yPos, zPos));
				uv.push_back(glm::vec2(xSegment, ySegment));
				normals.push_back(glm::vec3(xPos, yPos, zPos));
			}
		}

		bool oddRow = false;
		for (unsigned int y = 0; y < Y_SEGMENTS; ++y)
		{
			if (!oddRow) // even rows: y == 0, y == 2; and so on
			{
				for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
				{
					indices.push_back(y * (X_SEGMENTS + 1) + x);
					indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
				}
			}
			else
			{
				for (int x = X_SEGMENTS; x >= 0; --x)
				{
					indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
					indices.push_back(y * (X_SEGMENTS + 1) + x);
				}
			}
			oddRow = !oddRow;
		}
		indexCount = static_cast<unsigned int>(indices.size());

		std::vector<float> data;
		for (unsigned int i = 0; i < positions.size(); ++i)
		{
			data.push_back(positions[i].x);
			data.push_back(positions[i].y);
			data.push_back(positions[i].z);
			if (normals.size() > 0)
			{
				data.push_back(normals[i].x);
				data.push_back(normals[i].y);
				data.push_back(normals[i].z);
			}
			if (uv.size() > 0)
			{
				data.push_back(uv[i].x);
				data.push_back(uv[i].y);
			}
		}
		glBindVertexArray(sphereVAO);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
		unsigned int stride = (3 + 2 + 3) * sizeof(float);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
	}

	glBindVertexArray(sphereVAO);
	glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
}

void SetUniformMat4(GLuint shader_id, const char* uniform_name, glm::mat4 uniform_value)
{
	glUseProgram(shader_id);
	glUniformMatrix4fv(glGetUniformLocation(shader_id, uniform_name), 1, GL_FALSE, &uniform_value[0][0]);
}

void SetUniformVec3(GLuint shader_id, const char* uniform_name, glm::vec3 uniform_value)
{
	glUseProgram(shader_id);
	glUniform3fv(glGetUniformLocation(shader_id, uniform_name), 1, glm::value_ptr(uniform_value));
}

template <class T>
void SetUniform1Value(GLuint shader_id, const char* uniform_name, T uniform_value)
{
	glUseProgram(shader_id);
	glUniform1i(glGetUniformLocation(shader_id, uniform_name), uniform_value);
	glUseProgram(0);
}



const char* getVertexShaderSource()
{
	// For now, you use a string for your shader code, in the assignment, shaders will be stored in .glsl files
	return
		"#version 330 core\n"
		"layout (location = 0) in vec3 aPos;\n"
		"layout (location = 1) in vec3 aNormal;\n"
		"layout (location = 2) in vec2 aTexCoord;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view = mat4(1.0f);\n"  // default value for view matrix (identity)
		"uniform mat4 projection = mat4(1.0f);\n"
		"\n"
		"out vec3 Normal;\n" // normal vector at each vertex
		"out vec3 FragPos;\n" // fragment position
		"out vec2 TexCoord;\n"
		"\n"
		"void main()\n"
		"{\n"
		"gl_Position = projection * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0f);\n"
		"FragPos = vec3(model * vec4(aPos, 1.0));\n"
		"Normal = aNormal;\n"
		"TexCoord = aTexCoord;\n"
		"}\n\0";
}

const char* getFragmentShaderSource()
{
	return
		"#version 330 core\n"
		"uniform vec3 materialColor;\n"
		"uniform sampler2D textureSampler;\n"

		"out vec4 FragColor;\n"
		"in vec2 TexCoord;\n"
		"\n"
		"void main()"
		"{"
		"vec4 textureColor = texture(textureSampler, TexCoord);"
		"FragColor = textureColor * vec4(materialColor, 1.0f);"
		"}";
}

int compileAndLinkShaders()
{
	// compile and link shader program
	// return shader program id
	// ------------------------------------

	// vertex shader
	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	const char* vertexShaderSource = getVertexShaderSource();
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	// check for shader compile errors
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// fragment shader
	int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	const char* fragmentShaderSource = getFragmentShaderSource();
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	// check for shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// link shaders
	int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

void subdivide(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, int depth) {
	if (depth == 0) {
		sphereVertices.push_back(a);
		sphereVertices.push_back(b);
		sphereVertices.push_back(c);
		return;
	}

	glm::vec3 mid1 = glm::normalize(a + b);
	glm::vec3 mid2 = glm::normalize(b + c);
	glm::vec3 mid3 = glm::normalize(c + a);

	subdivide(a, mid1, mid3, depth - 1);
	subdivide(b, mid2, mid1, depth - 1);
	subdivide(c, mid3, mid2, depth - 1);
	subdivide(mid1, mid2, mid3, depth - 1);
}

void createSphere() {
	glm::vec3 v1 = glm::vec3(0.0, 1.0, 0.0);
	glm::vec3 v2 = glm::vec3(0.0, -1.0, 0.0);
	glm::vec3 v3 = glm::vec3(1.0, 0.0, 0.0);
	glm::vec3 v4 = glm::vec3(-1.0, 0.0, 0.0);
	glm::vec3 v5 = glm::vec3(0.0, 0.0, 1.0);
	glm::vec3 v6 = glm::vec3(0.0, 0.0, -1.0);

	subdivide(v1, v3, v5, 3);
	subdivide(v3, v2, v5, 3);
	subdivide(v2, v4, v5, 3);
	subdivide(v4, v1, v5, 3);
	subdivide(v1, v6, v3, 3);
	subdivide(v3, v6, v2, 3);
	subdivide(v2, v6, v4, 3);
	subdivide(v4, v6, v1, 3);
}

// Function to insert a 3D vector (three float values) into a given array
void insertVec3(std::vector<float>& array, float valx, float valy, float valz)
{
	// Add the x, y, z values to the end of the array
	array.push_back(valx); // Add x-coordinate
	array.push_back(valy); // Add y-coordinate
	array.push_back(valz); // Add z-coordinate
}

// Function to create a Net Mesh for 3D models
void prepareNet()
{
	// Define the dimensions and step size of the net
	float xsize = 7;
	float ysize = 7;
	float step = 1.0f; // Size of each small square in the net

	// Vector to store vertex data
	std::vector<float> vertices_value;

	// Loop over x and y values, insert vertices for each square in the net
	for (float x = -xsize / 2; x <= xsize / 2; x += step) {
		for (float z = -ysize / 2; z <= ysize / 2; z += step) {

			// Insert four vertices for each square in the net
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z + step);

			// Insert four vertices for each square in the net in reverse order
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x, 0.0f, z + step);
		}
	}

	// Compute the total number of vertices
	netSize = (int)vertices_value.size() / 3;

	// Generate unique IDs for Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &vaoNet);
	glGenBuffers(1, &vboNet);

	// Bind the Vertex Array Object
	glBindVertexArray(vaoNet);

	// Bind the Vertex Buffer Object and assign vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vboNet);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices_value.size(), vertices_value.data(), GL_STATIC_DRAW);

	// Specify the attributes of the vertices (position)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0); // Enable vertex attribute at location 0
}

void prepareBigNet()
{
	// Define the dimensions and step size of the net
	float xsize = 39;
	float ysize = 15;
	float step = 2.0f; // Size of each small square in the net

	// Vector to store vertex data
	std::vector<float> vertices_value;

	// Loop over x and y values, insert vertices for each square in the net
	for (float x = -xsize / 2; x <= xsize / 2; x += step) {
		for (float z = -ysize / 2; z <= ysize / 2; z += step) {

			// Insert four vertices for each square in the net
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z + step);

			// Insert four vertices for each square in the net in reverse order
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x, 0.0f, z + step);
		}
	}

	// Compute the total number of vertices
	netBigSize = (int)vertices_value.size() / 3;

	// Generate unique IDs for Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &vaoBigNet);
	glGenBuffers(1, &vboBigNet);

	// Bind the Vertex Array Object
	glBindVertexArray(vaoBigNet);

	// Bind the Vertex Buffer Object and assign vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vboBigNet);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices_value.size(), vertices_value.data(), GL_STATIC_DRAW);

	// Specify the attributes of the vertices (position)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0); // Enable vertex attribute at location 0
}


// Function to create a grid for the 3D space
void prepareGrid()
{
	// Define the dimensions and step size of the grid
	float size = 100.0f; // Size of the square grid
	float step = 1.0f; // Size of each small square in the grid

	// Vector to store vertex data
	std::vector<float> vertices_value;

	// Loop over x and z values, insert vertices for each square in the grid
	for (float x = -size / 2; x <= size / 2; x += step) {
		for (float z = -size / 2; z <= size / 2; z += step) {

			// Insert four vertices for each square in the grid
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z + step);

			// Insert four vertices for each square in the grid in reverse order
			insertVec3(vertices_value, x + step, 0.0f, z);
			insertVec3(vertices_value, x + step, 0.0f, z + step);
			insertVec3(vertices_value, x, 0.0f, z);
			insertVec3(vertices_value, x, 0.0f, z + step);
		}
	}

	// Compute the total number of vertices
	gridSize = (int)vertices_value.size() / 3;

	// Generate unique IDs for Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &vaoGrid);
	glGenBuffers(1, &vboGrid);

	// Bind the Vertex Array Object
	glBindVertexArray(vaoGrid);

	// Bind the Vertex Buffer Object and assign vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vboGrid);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices_value.size(), vertices_value.data(), GL_STATIC_DRAW);

	// Specify the attributes of the vertices (position)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0); // Enable vertex attribute at location 0
}


// Function to create axes for the 3D space
void prepareAxes()
{
	// Vector to store vertex data for the axes
	std::vector<float> vertices_value;

	// Define vertices for the x-axis
	insertVec3(vertices_value, 0, 0.01, 0);
	insertVec3(vertices_value, 5, 0.01, 0);

	// Define vertices for the y-axis
	insertVec3(vertices_value, 0, 0.01, 0);
	insertVec3(vertices_value, 0, 5.01, 0);

	// Define vertices for the z-axis
	insertVec3(vertices_value, 0, 0.01, 0);
	insertVec3(vertices_value, 0, 0.01, 5);

	// Generate unique IDs for Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &vaoAxes);
	glGenBuffers(1, &vboAxes);

	// Bind the Vertex Array Object
	glBindVertexArray(vaoAxes);

	// Bind the Vertex Buffer Object and assign vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vboAxes);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices_value.size(), vertices_value.data(), GL_STATIC_DRAW);

	// Specify the attributes of the vertices (position)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0); // Enable vertex attribute at location 0
}


// Function to create a cube in the 3D space
void prepareCube()
{
	// Define the vertices for the cube
	float vertices[] = {
		// back face
				-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 0.0f, // bottom-left
				 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 1.0f, // top-right
				 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 0.0f, // bottom-right
				 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 1.0f, // top-right
				-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 0.0f, // bottom-left
				-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 1.0f, // top-left
				// front face
				-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 0.0f, // bottom-left
				 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 0.0f, // bottom-right
				 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 1.0f, // top-right
				 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 1.0f, // top-right
				-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 1.0f, // top-left
				-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 0.0f, // bottom-left
				// left face
				-0.5f,  0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
				-0.5f,  0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
				-0.5f, -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
				-0.5f, -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
				-0.5f, -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
				-0.5f,  0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
				// right face
				 0.5f,  0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
				 0.5f, -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
				 0.5f,  0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right
				 0.5f, -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
				 0.5f,  0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
				 0.5f, -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left
				 // bottom face
				 -0.5f, -0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 1.0f, // top-right
				  0.5f, -0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 1.0f, // top-left
				  0.5f, -0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f, // bottom-left
				  0.5f, -0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f, // bottom-left
				 -0.5f, -0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 0.0f, // bottom-right
				 -0.5f, -0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 1.0f, // top-right
				 // top face
				 -0.5f,  0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 1.0f, // top-left
				  0.5f,  0.5f , 0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f, // bottom-right
				  0.5f,  0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 1.0f, // top-right
				  0.5f,  0.5f,  0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f, // bottom-right
				 -0.5f,  0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 1.0f, // top-left
				 -0.5f,  0.5f,  0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 0.0f  // bottom-left
	};

	// Generate unique IDs for Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &vaoCube);
	glGenBuffers(1, &vboCube);

	// Bind the Vertex Buffer Object and assign vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vboCube);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Bind the Vertex Array Object
	glBindVertexArray(vboCube);

	// Specify the attributes of the vertices (position)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
}



// Function to create a cube in the 3D space
void prepareSkybox()
{
	float skyArray[] = {

		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 1.0f, 1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -0.5f, 0.0f, 1.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 1.0f, 1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  0.5f, 0.0f, 0.0f,

		 0.5f, -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f,
		 0.5f, -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 1.0f,
		-0.5f, -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 1.0f,
		-0.5f, -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 0.0f, 0.0f,
		 0.5f, -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f,

		 0.5f,  0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 0.0f,
		-0.5f,  0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f,
		 0.5f,  0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f,
		 0.5f,  0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f, 0.0f,
		-0.5f,  0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -0.5f,  0.0f, 0.0f, 1.0f,

		 0.5f, -0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 1.0f,
		 0.5f,  0.5f , 0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.5f,  0.0f, 0.0f, 0.0f
	};

	glGenVertexArrays(1, &vaoSkybox);
	glGenBuffers(1, &vboSkybox);
	// fill buffer
	glBindBuffer(GL_ARRAY_BUFFER, vboSkybox);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyArray), skyArray, GL_STATIC_DRAW);
	// link vertex attributes
	glBindVertexArray(vaoSkybox);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
// Function to initialize OpenGL properties
void initialGL()
{
	// Enable depth testing to ensure correct z-ordering of our fragments
	glEnable(GL_DEPTH_TEST);

	// Disable face culling. Face culling is a process which tells OpenGL not 
	// to render faces of the object which are not visible to the viewer.
	glDisable(GL_CULL_FACE);
	// Load The Texture(s)
	glEnable(GL_TEXTURE_2D);
	// Prepare the vertex array objects for the grid, axes, cube, and net
	prepareGrid();
	prepareAxes();
	prepareCube();
	prepareSkybox();
	prepareNet();
	prepareBigNet();
	tattoo = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\tattoo.jpg");
	metalTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\metal.jpg");
	groundTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\clay.jpg");
	skyTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\sky.jpg");
	ballTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\ball.jpg");
	baseTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\default.jpg");
	blackTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\blackRock.jpg");
	woodTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\wood.jpg");
	rockTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\rock.jpg");
	techTexture = loadTexture("C:\\Users\\yulon\\Documents\\Texture\\tech.jpg");
	// Compile and link the shaders. Assign the resulting shader program ID to 'shaderProgram'
	shaderProgram = compileAndLinkShaders();
}

void drawSphere(glm::vec3 color, glm::mat4 mat) {
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(mat));
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(color));

	renderSphere();
}

// Function to draw the net with a specified color and model matrix
void drawNet(glm::vec3 color, glm::mat4 mat)
{
	// Pass the model matrix to the shader
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(mat));

	// Pass the color to the shader
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(color));

	// Bind the vertex array object for the net
	glBindVertexArray(vaoNet);

	// Draw the net as lines
	glDrawArrays(GL_LINES, 0, netSize);
}

void drawBigNet(glm::vec3 color, glm::mat4 mat)
{
	// Pass the model matrix to the shader
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(mat));

	// Pass the color to the shader
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(color));

	// Bind the vertex array object for the net
	glBindVertexArray(vaoBigNet);

	// Draw the net as lines
	glDrawArrays(GL_LINES, 0, netBigSize);
}

// Function to draw a cube with a specified color and model matrix
void drawCube(glm::vec3 color, glm::mat4 mat)
{
	// Bind the vertex array object for the cube
	glBindVertexArray(vaoCube);

	// Pass the model matrix to the shader
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(mat));

	// Pass the color to the shader
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(color));

	// Draw the cube as triangles. There are 36 vertices that make up the cube.
	glDrawArrays(GL_TRIANGLES, 0, 36);
}
// Function to draw a cube with a specified color and model matrix
void drawSkybox(glm::vec3 color, glm::mat4 mat)
{
	// Bind the vertex array object for the cube
	glBindVertexArray(vaoSkybox);

	// Pass the model matrix to the shader
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(mat));

	// Pass the color to the shader
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(glm::vec3(1.0f)));


	// Draw the cube as triangles. There are 36 vertices that make up the cube.

	glDrawArrays(GL_TRIANGLES, 0, 36);

}
void drawModel(glm::mat4 worldMat)
{

	glm::mat4 armMat_0(1.f);
	glm::mat4 trans_m_0 = glm::translate(glm::mat4(1.f), position_model);
	glm::mat4 modelRX = glm::rotate(glm::mat4(1.f), glm::radians(rotation_model.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY = glm::rotate(glm::mat4(1.f), glm::radians(rotation_model.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ = glm::rotate(glm::mat4(1.f), glm::radians(rotation_model.z), glm::vec3(0, 0, 1));
	glm::mat4 modelS = glm::scale(glm::mat4(1.f), glm::vec3(scale_value, scale_value, scale_value));


	armMat_0 = worldMat * trans_m_0 * modelS * modelRX * modelRY * modelRZ;
	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, tattoo);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawCube(glm::vec3(1.0, 1.0, 1.0), armMat_0 * glm::scale(glm::mat4(1.f), glm::vec3(2, 2, 6)));

	// Arm 1
	glm::mat4 armMat_1(1.f);
	glm::mat4 trans_m_1 = glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -2));
	glm::mat4 rotate_m_1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv2.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv2.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv2.z), glm::vec3(0, 0, 1));
	armMat_1 = armMat_0 * trans_m_1 * rotate_m_1 * modelRY1 * modelRZ1;

	glm::mat4 scale_m_1 = glm::scale(glm::mat4(1.f), glm::vec3(2, 4, 2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), armMat_1 * scale_m_1);
	glBindTexture(GL_TEXTURE_2D, 0);
	// racket

	// handle
	glm::mat4 racket_hand(1.f);
	glm::mat4 trans_m_hand = glm::translate(glm::mat4(1.f), glm::vec3(0, 4.5, 0));
	glm::mat4 rotate_m_hand = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv3.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY2 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv3.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ2 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelv3.z), glm::vec3(0, 0, 1));
	racket_hand = armMat_1 * trans_m_hand * rotate_m_hand * modelRY2 * modelRZ2;

	glm::mat4 scale_m_h = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 11, 1.2));
	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, metalTexture);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_hand * scale_m_h);

	// buttom
	glm::mat4 racket_buttom(1.f);
	glm::mat4 trans_m_b = glm::translate(glm::mat4(1.f), glm::vec3(0, 6, 0));

	racket_buttom = racket_hand * trans_m_b;
	glm::mat4 scale_m_b = glm::scale(glm::mat4(1.f), glm::vec3(8, 1.2, 1.2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_buttom * scale_m_b);

	// top
	glm::mat4 racket_top(1.f);
	glm::mat4 trans_m_t = glm::translate(glm::mat4(1.f), glm::vec3(0, 16, 0));

	racket_top = racket_hand * trans_m_t;
	glm::mat4 scale_m_t = glm::scale(glm::mat4(1.f), glm::vec3(8, 1.2, 1.2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_top * scale_m_t);



	// right
	glm::mat4 racket_right(1.f);
	glm::mat4 trans_m_r = glm::translate(glm::mat4(1.f), glm::vec3(4, 11, 0));

	racket_right = racket_hand * trans_m_r;
	glm::mat4 scale_m_r = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 10, 1.2));

	drawCube(glm::vec3(1.0, 0.0, 0.0), racket_right * scale_m_r);

	// left
	glm::mat4 racket_left(1.f);
	glm::mat4 trans_m_l = glm::translate(glm::mat4(1.f), glm::vec3(-4, 11, 0));

	racket_left = racket_hand * trans_m_l;
	glm::mat4 scale_m_l = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 10, 1.2));
	drawCube(glm::vec3(1.0, 0.0, 0.0), racket_left * scale_m_l);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_2D, baseTexture);
	// net
	glm::mat4 racket_net(1.f);
	glm::mat4 trans_m_c = glm::translate(glm::mat4(1.f), glm::vec3(0, 11, 0));
	glm::mat4 rotate_m_c = glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1, 0, 0));
	racket_net = racket_hand * trans_m_c;
	drawNet(glm::vec3(0.0, 0.0, 0.0), racket_net * rotate_m_c);

	// Y shape
	glm::vec3 yColor = glm::vec3(1.0, 1.0, 1.0); // change this to your desired color
	float yScale = 1.0; // change this to your desired scale

	// Y stem
	glm::mat4 yStem(1.f);
	glm::mat4 trans_yStem = glm::translate(glm::mat4(1.f), position_model + glm::vec3(0, 24, 11));
	glm::mat4 scale_yStem = glm::scale(glm::mat4(1.f), glm::vec3(1, 4, 1) * yScale);
	yStem = worldMat * trans_yStem * scale_yStem;
	glBindTexture(GL_TEXTURE_2D, techTexture);
	drawCube(yColor, yStem);

	// Y left branch
	glm::mat4 yLeft(1.f);
	glm::mat4 trans_yLeft = glm::translate(glm::mat4(1.f), position_model + glm::vec3(-1, 27, 11));
	glm::mat4 rotate_yLeft = glm::rotate(glm::mat4(1.f), glm::radians(45.f), glm::vec3(0, 0, 1));
	glm::mat4 scale_yLeft = glm::scale(glm::mat4(1.f), glm::vec3(1, 4, 1) * yScale);
	yLeft = worldMat * trans_yLeft * rotate_yLeft * scale_yLeft;
	drawCube(yColor, yLeft);

	// Y right branch
	glm::mat4 yRight(1.f);
	glm::mat4 trans_yRight = glm::translate(glm::mat4(1.f), position_model + glm::vec3(1, 27, 11));
	glm::mat4 rotate_yRight = glm::rotate(glm::mat4(1.f), glm::radians(-45.f), glm::vec3(0, 0, 1));
	glm::mat4 scale_yRight = glm::scale(glm::mat4(1.f), glm::vec3(1, 4, 1) * yScale);
	yRight = worldMat * trans_yRight * rotate_yRight * scale_yRight;
	drawCube(yColor, yRight);
	glBindTexture(GL_TEXTURE_2D, 0);

	

	// Left bar of 'A'
	glm::mat4 aMatLeft(1.f);
	glm::mat4 trans_a_left = glm::translate(glm::mat4(1.f), position_model + glm::vec3(2, 26, 7)); // You can adjust the height (20 here)
	glm::mat4 rotate_a_left = glm::rotate(glm::mat4(1.f), glm::radians(30.f), glm::vec3(0, 0, 1));
	aMatLeft = worldMat * trans_a_left * rotate_a_left;
	glm::mat4 scale_a_left = glm::scale(glm::mat4(1.f), glm::vec3(0.5, 8, 1)); // Adjust the scale for the shape of 'A'
	glBindTexture(GL_TEXTURE_2D, woodTexture);
	drawCube(glm::vec3(1.0, 0.0, 0.0), aMatLeft * scale_a_left);

	// Right bar of 'A'
	glm::mat4 aMatRight(1.f);
	glm::mat4 trans_a_right = glm::translate(glm::mat4(1.f), position_model + glm::vec3(-2, 26, 7)); // You can adjust the height (20 here)
	glm::mat4 rotate_a_right = glm::rotate(glm::mat4(1.f), glm::radians(-30.f), glm::vec3(0, 0, 1));
	aMatRight = worldMat * trans_a_right * rotate_a_right;
	glm::mat4 scale_a_right = glm::scale(glm::mat4(1.f), glm::vec3(0.5, 8, 1)); // Adjust the scale for the shape of 'A'
	drawCube(glm::vec3(1.0, 0.0, 0.0), aMatRight * scale_a_right);

	// Middle bar of 'A'
	glm::mat4 aMatMiddle(1.f);
	glm::mat4 trans_a_middle = glm::translate(glm::mat4(1.f), position_model + glm::vec3(0, 26, 7)); // You can adjust the height (24 here)
	aMatMiddle = trans_a_middle;
	glm::mat4 scale_a_middle = glm::scale(glm::mat4(1.f), glm::vec3(4, 0.5, 1)); // Adjust the scale for the shape of 'A'
	drawCube(glm::vec3(1.0, 0.0, 0.0), aMatMiddle * scale_a_middle);
	glBindTexture(GL_TEXTURE_2D, 0);

	glm::mat4 ball(1.f);
	glm::mat4 trans_ball = glm::translate(glm::mat4(1.f), glm::vec3(0, 18, 0)); // You can adjust the height (24 here)
	glm::mat4 rotate_ball = glm::rotate(glm::mat4(1.f), glm::radians(0.f), glm::vec3(1, 0, 0));
	ball = armMat_0 * trans_ball * rotate_ball;
	glm::mat4 scale_ball = glm::scale(glm::mat4(1.f), glm::vec3(-1.0f, 1.0f, 1.0f)); // Adjust the scale for the shape of 'A'

	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, ballTexture);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawSphere(glm::vec3(1.0, 1.0, 1.0), ball * scale_ball);
	glBindTexture(GL_TEXTURE_2D, 0);



}

void drawModelP2(glm::mat4 worldMat)
{
	// The model is made of several parts: 2 arms, a hand, and a racket (including a top, bottom, left, and right part). 

	// Arm 0
	// Apply transformations: translation (to move the model in the 3D space), 
	// scaling (to resize the model), and rotation (to rotate the model around the x, y, and z axes).
	// Draw a cube for the arm with a specified color and transformation.

	glm::mat4 armMat_0(1.f);
	glm::mat4 trans_m_0 = glm::translate(glm::mat4(1.f), position_modelP2);
	glm::mat4 modelRX = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2.z), glm::vec3(0, 0, 1));
	glm::mat4 modelS = glm::scale(glm::mat4(1.f), glm::vec3(scale_value, scale_value, scale_value));


	armMat_0 = worldMat * trans_m_0 * modelS * modelRX * modelRY * modelRZ;
	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, tattoo);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawCube(glm::vec3(1.0, 1.0, 1.0), armMat_0 * glm::scale(glm::mat4(1.f), glm::vec3(2, 2, 6)));

	// Arm 1
	glm::mat4 armMat_1(1.f);
	glm::mat4 trans_m_1 = glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -2));
	glm::mat4 rotate_m_1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v2.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v2.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v2.z), glm::vec3(0, 0, 1));
	armMat_1 = armMat_0 * trans_m_1 * rotate_m_1 * modelRY1 * modelRZ1;

	glm::mat4 scale_m_1 = glm::scale(glm::mat4(1.f), glm::vec3(2, 4, 2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), armMat_1 * scale_m_1);
	glBindTexture(GL_TEXTURE_2D, 0);
	// racket

	// handle
	glm::mat4 racket_hand(1.f);
	glm::mat4 trans_m_hand = glm::translate(glm::mat4(1.f), glm::vec3(0, 4.5, 0));
	glm::mat4 rotate_m_hand = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v3.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY2 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v3.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ2 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_modelP2v3.z), glm::vec3(0, 0, 1));
	racket_hand = armMat_1 * trans_m_hand * rotate_m_hand * modelRY2 * modelRZ2;

	glm::mat4 scale_m_h = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 11, 1.2));
	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, metalTexture);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_hand * scale_m_h);

	// buttom
	glm::mat4 racket_buttom(1.f);
	glm::mat4 trans_m_b = glm::translate(glm::mat4(1.f), glm::vec3(0, 6, 0));

	racket_buttom = racket_hand * trans_m_b;
	glm::mat4 scale_m_b = glm::scale(glm::mat4(1.f), glm::vec3(8, 1.2, 1.2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_buttom * scale_m_b);

	// top
	glm::mat4 racket_top(1.f);
	glm::mat4 trans_m_t = glm::translate(glm::mat4(1.f), glm::vec3(0, 16, 0));

	racket_top = racket_hand * trans_m_t;
	glm::mat4 scale_m_t = glm::scale(glm::mat4(1.f), glm::vec3(8, 1.2, 1.2));
	drawCube(glm::vec3(1.0, 1.0, 1.0), racket_top * scale_m_t);



	// right
	glm::mat4 racket_right(1.f);
	glm::mat4 trans_m_r = glm::translate(glm::mat4(1.f), glm::vec3(4, 11, 0));

	racket_right = racket_hand * trans_m_r;
	glm::mat4 scale_m_r = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 10, 1.2));

	drawCube(glm::vec3(1.0, 0.0, 0.0), racket_right * scale_m_r);

	// left
	glm::mat4 racket_left(1.f);
	glm::mat4 trans_m_l = glm::translate(glm::mat4(1.f), glm::vec3(-4, 11, 0));

	racket_left = racket_hand * trans_m_l;
	glm::mat4 scale_m_l = glm::scale(glm::mat4(1.f), glm::vec3(1.2, 10, 1.2));
	drawCube(glm::vec3(1.0, 0.0, 0.0), racket_left * scale_m_l);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_2D, baseTexture);
	// net
	glm::mat4 racket_net(1.f);
	glm::mat4 trans_m_c = glm::translate(glm::mat4(1.f), glm::vec3(0, 11, 0));
	glm::mat4 rotate_m_c = glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1, 0, 0));
	racket_net = racket_hand * trans_m_c;
	drawNet(glm::vec3(0.0, 0.0, 0.0), racket_net * rotate_m_c);

	// Draw N
	// Left part of 'N'
	glm::mat4 nMatLeft(1.f);
	glm::mat4 trans_n_left = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(-2, 24, -11)); // Modify the height (20 here) as needed
	nMatLeft = worldMat * trans_n_left;
	glm::mat4 scale_n_left = glm::scale(glm::mat4(1.f), glm::vec3(1, 8, 1)); // Adjust the scale for the shape of 'N'
	glBindTexture(GL_TEXTURE_2D, rockTexture);
	drawCube(glm::vec3(1.f, 0.f, 0.f), nMatLeft * scale_n_left);

	// Right part of 'N'
	glm::mat4 nMatRight(1.f);
	glm::mat4 trans_n_right = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(2, 24, -11)); // Modify the height (20 here) as needed
	nMatRight = worldMat * trans_n_right;
	glm::mat4 scale_n_right = glm::scale(glm::mat4(1.f), glm::vec3(1, 8, 1)); // Adjust the scale for the shape of 'N'
	drawCube(glm::vec3(1.f, 0.f, 0.f), nMatRight * scale_n_right);

	// Middle part of 'N'
	glm::mat4 nMatMiddle(1.f);
	glm::mat4 trans_n_middle = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(0, 23.75, -11)); // Modify the height (20 here) as needed
	glm::mat4 rotate_n_middle = glm::rotate(glm::mat4(1.f), glm::radians(30.f), glm::vec3(0, 0, 1));
	nMatMiddle = worldMat * trans_n_middle * rotate_n_middle;
	glm::mat4 scale_n_middle = glm::scale(glm::mat4(1.f), glm::vec3(1, 8, 1)); // Adjust the scale for the shape of 'N'
	drawCube(glm::vec3(1.f, 0.f, 0.f), nMatMiddle * scale_n_middle);
	glBindTexture(GL_TEXTURE_2D, 0);
	// Draw G
		// Left part of 'G'
	glm::mat4 gMatLeft(1.f);
	glm::mat4 trans_g_left = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(-2, 24, -7)); // Adjust the height (20 here) as needed
	gMatLeft = worldMat * trans_g_left;
	glm::mat4 scale_g_left = glm::scale(glm::mat4(1.f), glm::vec3(1, 8, 1)); // Adjust the scale for the shape of 'G'
	glBindTexture(GL_TEXTURE_2D, blackTexture);
	drawCube(glm::vec3(1.0, 1.0, 1.0), gMatLeft * scale_g_left);

	// Top part of 'G'
	glm::mat4 gMatTop(1.f);
	glm::mat4 trans_g_top = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(0, 28, -7)); // Adjust the height (24 here) as needed
	gMatTop = worldMat * trans_g_top;
	glm::mat4 scale_g_top = glm::scale(glm::mat4(1.f), glm::vec3(4, 1, 1)); // Adjust the scale for the shape of 'G'
	drawCube(glm::vec3(1.0, 1.0, 1.0), gMatTop * scale_g_top);

	// Bottom part of 'G'
	glm::mat4 gMatBottom(1.f);
	glm::mat4 trans_g_bottom = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(0, 20, -7)); // Adjust the height (16 here) as needed
	gMatBottom = worldMat * trans_g_bottom;
	glm::mat4 scale_g_bottom = glm::scale(glm::mat4(1.f), glm::vec3(4, 1, 1)); // Adjust the scale for the shape of 'G'
	drawCube(glm::vec3(1.0, 1.0, 1.0), gMatBottom * scale_g_bottom);

	// Right part of 'G'
	glm::mat4 gMatRight(1.f);
	glm::mat4 trans_g_right = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(2, 21.5, -7)); // Adjust the height (22 here) as needed
	gMatRight = worldMat * trans_g_right;
	glm::mat4 scale_g_right = glm::scale(glm::mat4(1.f), glm::vec3(1, 2, 1)); // Adjust the scale for the shape of 'G'
	drawCube(glm::vec3(1.0, 1.0, 1.0), gMatRight * scale_g_right);

	// Inner part of 'G'
	glm::mat4 gMatInner(1.f);
	glm::mat4 trans_g_inner = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(1, 23, -7)); // Adjust the height (18 here) as needed
	gMatInner = worldMat * trans_g_inner;
	glm::mat4 scale_g_inner = glm::scale(glm::mat4(1.f), glm::vec3(1, 1, 1)); // Adjust the scale for the shape of 'G'
	drawCube(glm::vec3(1.0, 1.0, 1.0), gMatInner * scale_g_inner);
	glBindTexture(GL_TEXTURE_2D, 0);

	glm::mat4 ball(1.f);
	glm::mat4 trans_ball = glm::translate(glm::mat4(1.f), position_modelP2 + glm::vec3(0, 20, -30)); // You can adjust the height (24 here)
	ball = armMat_0 * trans_ball;
	glm::mat4 scale_ball = glm::scale(glm::mat4(1.f), glm::vec3(-1.0f, 1.0f, 1.0f));

	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, ballTexture);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawSphere(glm::vec3(1, 1, 1), ball * scale_ball);
	glBindTexture(GL_TEXTURE_2D, 0);
}


void drawDaNet(glm::mat4 worldMat)
{


	glm::mat4 pillar_0(1.f);
	glm::mat4 trans_m_0 = glm::translate(glm::mat4(1.f), glm::vec3(0, 7, 0));
	glm::mat4 modelRX = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.x), glm::vec3(1, 0, 0));
	glm::mat4 modelRY = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.y), glm::vec3(0, 1, 0));
	glm::mat4 modelRZ = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.z), glm::vec3(0, 0, 1));



	pillar_0 = worldMat * trans_m_0 * modelRX * modelRY * modelRZ;
	glBindTexture(GL_TEXTURE_2D, groundTexture);
	drawCube(glm::vec3(0.0, 0.0, 0.0), pillar_0 * glm::scale(glm::mat4(1.f), glm::vec3(1, 15, 1)));

	// Pillar 1
	glm::mat4 pillar_1(1.f);
	glm::mat4 trans_m_1 = glm::translate(glm::mat4(1.f), glm::vec3(-19.5, 0, 0));
	glm::mat4 rotate_m_1 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.x), glm::vec3(1, 0, 0));
	pillar_1 = pillar_0 * trans_m_1 * rotate_m_1;

	drawCube(glm::vec3(0.0, 0.0, 0.0), pillar_1 * glm::scale(glm::mat4(1.f), glm::vec3(1, 15, 1)));

	// Pillar 2
	glm::mat4 pillar_2(1.f);
	glm::mat4 trans_m_2 = glm::translate(glm::mat4(1.f), glm::vec3(20, 0, 0));
	glm::mat4 rotate_m_2 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.x), glm::vec3(1, 0, 0));
	pillar_2 = pillar_0 * trans_m_2 * rotate_m_2;


	drawCube(glm::vec3(0.0, 0.0, 0.0), pillar_2 * glm::scale(glm::mat4(1.f), glm::vec3(1, 15, 1)));

	// Pillar 3
	glm::mat4 pillar_3(1.f);
	glm::mat4 trans_m_3 = glm::translate(glm::mat4(1.f), glm::vec3(0, 8, 0));
	glm::mat4 rotate_m_3 = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.x), glm::vec3(1, 0, 0));
	pillar_2 = pillar_0 * trans_m_3 * rotate_m_3;


	drawCube(glm::vec3(1.0, 1.0, 1.0), pillar_2 * glm::scale(glm::mat4(1.f), glm::vec3(41, 2, 2)));

	// net
	glm::mat4 net(1.f);
	glm::mat4 trans_m_c = glm::translate(glm::mat4(1.f), glm::vec3(0, 1, 0));
	glm::mat4 rotate_m_c = glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1, 0, 0));
	net = pillar_0 * trans_m_c;
	drawBigNet(glm::vec3(0.0, 0.0, 0.0), net * rotate_m_c);
	glBindTexture(GL_TEXTURE_2D, 0);
	// floor
	glm::mat4 floor(1.f);
	glm::mat4 trans_m_floor = glm::translate(glm::mat4(1.f), glm::vec3(0, -0.95, 0));
	glm::mat4 rotate_m_floor = glm::rotate(glm::mat4(1.f), glm::radians(rotation_net.x), glm::vec3(1, 0, 0));
	floor = worldMat * trans_m_floor * rotate_m_floor;

	if (textureSwitch) {
		glBindTexture(GL_TEXTURE_2D, groundTexture);
	}
	else {
		glBindTexture(GL_TEXTURE_2D, baseTexture);
	}
	drawCube(glm::vec3(greenColor), floor * glm::scale(glm::mat4(1.f), glm::vec3(36, 2, 78)));
	glBindTexture(GL_TEXTURE_2D, 0);
}

//function is used to set the render mode for the current frame, based on the value of the global renderMode variable.
void drawFrame()
{
	// Draw geometry
	switch (renderMode)
	{
	case 0:
		//The GL_FRONT_AND_BACK parameter tells OpenGL to apply this mode to both front and back faces of polygons.
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); //This line sets the polygon rasterization mode to point. In this mode, only vertices of polygons are drawn as points.
		break;
	case 1:
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // This line sets the polygon rasterization mode to line. In this mode, only the edges of polygons are drawn as lines, creating a wireframe view.
		break;
	case 2:
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); //This line sets the polygon rasterization mode to fill. In this mode, polygons are completely filled.
		break;
	default:
		break;
	}


	// world matrix
	//The code here first sets up a world transformation matrix that incorporates the current position and rotation of the world, then it applies this transformation to the grid.
	glm::mat4 worldMat(1.f);
	glm::mat4 worldT = glm::translate(glm::mat4(1.f), position_world);
	glm::mat4 worldRX = glm::rotate(glm::mat4(1.f), glm::radians(rotation_world.x), glm::vec3(1, 0, 0));
	glm::mat4 worldRY = glm::rotate(glm::mat4(1.f), glm::radians(rotation_world.y), glm::vec3(0, 1, 0));
	glm::mat4 worldRZ = glm::rotate(glm::mat4(1.f), glm::radians(rotation_world.z), glm::vec3(0, 0, 1));

	worldMat = worldT * worldRX * worldRY * worldRZ;

	// Grid
	glLineWidth(1.f);
	glUseProgram(shaderProgram);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(worldMat));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMat));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projMat));
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(yellowColor));

	glBindVertexArray(vaoGrid);
	glBindTexture(GL_TEXTURE_2D, baseTexture);
	glDrawArrays(GL_LINES, 0, gridSize);


	// Axes
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_LINE_WIDTH);
	glLineWidth(4.5f);
	glBindVertexArray(vaoAxes);
	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(redColor));
	glDrawArrays(GL_LINES, 0, 2);

	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(greenColor));
	glDrawArrays(GL_LINES, 2, 2);

	glUniform3fv(glGetUniformLocation(shaderProgram, "materialColor"), 1, glm::value_ptr(blueColor));
	glDrawArrays(GL_LINES, 4, 2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_LINE_WIDTH);

	// Model
	drawModel(worldMat);

	drawModelP2(worldMat);

	drawDaNet(worldMat);

	glm::mat4 sky(1.f);

	glm::mat4 trans_sky = glm::translate(glm::mat4(1.f), glm::vec3(0, 0, 0));
	glm::mat4 scale_sky = glm::scale(glm::mat4(1.f), glm::vec3(200, 200, 200));
	sky = worldMat * trans_sky * scale_sky;

	glBindTexture(GL_TEXTURE_2D, skyTexture);
	drawSkybox(glm::vec3(0.0f, 0.8f, 1.0f), sky * glm::scale(glm::mat4(1.f), glm::vec3(1, 1, 1)));
	glBindTexture(GL_TEXTURE_2D, 0);
	// End Frame
}


int main(int argc, char* argv[])
{
	// Initialize GLFW and OpenGL version
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create Window and rendering context using GLFW, resolution is 800x600
	GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Comp371 - Assignment 1", NULL, NULL);
	if (window == NULL)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// Set up mouse movement callback
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scroll_callback);
	// Enable cursor capture (optional)
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set up mouse movement callback

	projMat = glm::perspective(glm::radians(60.f), screenWidth / (float)screenHeight, 0.1f, 1000.f);


	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to create GLEW" << std::endl;
		glfwTerminate();
		return -1;
	}

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);




	initialGL();



	// Entering Main Loop
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		key_callback(window);
		// Each frame, reset color of each pixel to glClearColor
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glActiveTexture(GL_TEXTURE0);
		GLuint textureLocation = glGetUniformLocation(shaderProgram, "textureSampler");

		glUniform1i(textureLocation, 0);


		drawFrame();




		viewMat = camera.GetViewMatrix();




		glfwSwapBuffers(window);
		glfwPollEvents();

	}

	// Shutdown GLFW
	glfwTerminate();

	return 0;
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouseCallback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void key_callback(GLFWwindow* window)
{
	positionArray.push_back(&position_model);
	positionArray.push_back(&position_modelP2);

	rotationArray.push_back(&rotation_model);
	rotationArray.push_back(&rotation_modelv2);
	rotationArray.push_back(&rotation_modelv3);
	rotationArray.push_back(&rotation_modelP2);
	rotationArray.push_back(&rotation_modelP2v2);
	rotationArray.push_back(&rotation_modelP2v3);
	int currentSpaceState = glfwGetKey(window, GLFW_KEY_SPACE);

	// Close the window when ESC is pressed
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);



	// Check if the key is pressed and was not pressed in the previous iteration
	if (currentSpaceState == GLFW_PRESS && !spacePressed) {
		spacePressed = true;
		spacePositionUpdated = false;
	}

	// Check if the key was released
	if (currentSpaceState == GLFW_RELEASE && spacePressed) {
		spacePressed = false;
	}


	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		if (!onePressed) {
			player = 0;
			onePressed = true;
		}
		else {
			onePressed = false;
		}
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		if (!twoPressed) {
			player = 1;
			twoPressed = true;
		}
		else {
			twoPressed = false;
		}
	}
	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
		if (!twoPressed) {
			player = 2;
			threePressed = true;
		}
		else {
			threePressed = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
		if (!twoPressed) {
			player = 3;
			fourPressed = true;
		}
		else {
			fourPressed = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
		if (!twoPressed) {
			player = 4;
			fivePressed = true;
		}
		else {
			fivePressed = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
		if (!twoPressed) {
			player = 5;
			sixPressed = true;
		}
		else {
			sixPressed = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		if (spacePressed && !spacePositionUpdated) {
			glm::vec3* position = positionArray[player];
			(*position) = glm::linearRand(glm::vec3(-50.0f), glm::vec3(50.0f));
			(*position).y = 0.5f;
			spacePositionUpdated = true;
		}

	}

	int currentXState = glfwGetKey(window, GLFW_KEY_X);
	if (currentXState == GLFW_PRESS && !xPressed) {

		if (textureSwitch) {
			textureSwitch = false;

		}
		else {
			textureSwitch = true;
		}
		xPressed = true;
		// Perform any actions you want when X is pressed
	}

	// Check if the key was released
	if (currentXState == GLFW_RELEASE && xPressed) {

		xPressed = false;
		// Perform any actions you want when X is released
	}


	// Increase scale of the model when 'U' is pressed
	if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
		scale_value += 0.001;

	// Decrease scale of the model when 'J' is pressed
	if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
		scale_value -= 0.001;

	// Move the model in positive X direction when 'W' is pressed
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		glm::vec3* position = positionArray[player];
		(*position).x += 0.02;
	}

	// Move the model in positive Z direction when 'A' is pressed
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		glm::vec3* position = positionArray[player];
		(*position).z += 0.05;
	}

	// Move the model in negative X direction when 'S' is pressed
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		glm::vec3* position = positionArray[player];
		(*position).x -= 0.05;
	}

	// Move the model in negative Z direction when 'D' is pressed
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		glm::vec3* position = positionArray[player];
		(*position).z -= 0.05;
	}

	// Rotate the model in negative X direction when 'Z' is pressed
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).x -= 0.05;
	}
	// Rotate the model in positive X direction when 'C' is pressed
	if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).x += 0.05;
	}

	// Rotate the model in negative Y direction when 'Q' is pressed
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).y -= 0.05;
	}
	// Rotate the model in positive Y direction when 'E' is pressed
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).y += 0.05;
	}

	// Rotate the model in negative Z direction when 'R' is pressed
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).z -= 0.05;
	}
	// Rotate the model in positive Z direction when 'T' is pressed
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
		glm::vec3* rotation = rotationArray[player];
		(*rotation).z += 0.05;
	}


	// Set render mode to point when 'P' is pressed
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		if (!pPressed) {
			renderMode = 0;
			pPressed = true;
		}
		else {
			pPressed = false;
		}
	}


	// Set render mode to line when 'L' is pressed
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
		if (!lPressed) {
			renderMode = 1;
			lPressed = true;
		}
		else {
			lPressed = false;
		}
	}

	// Set render mode to fill (triangle) when 'h' is pressed
	if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
		if (!hPressed) {
			renderMode = 2;
			hPressed = true;
		}
		else {
			hPressed = false;
		}
	}

	// Rotate the world in positive X direction when RIGHT arrow key is pressed
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		rotation_world.x += 0.05f;

	// Rotate the world in positive Y direction when UP arrow key is pressed
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		rotation_world.y += 0.05f;

	// Rotate the world in positive Z direction when DOWN arrow key is pressed
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		rotation_world.z += 0.05f;

	// Rotate the world in negative X direction when LEFT arrow key is pressed
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		rotation_world.x -= 0.05f;

	//Camera movement
	if (glfwGetKey(window, GLFW_KEY_KP_8) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_KP_5) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_KP_4) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_KP_6) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
		if (!mPressed) {
			if (cameraSwitch == 0) {
				camera.Position = glm::vec3(0.0f, 40.0f, 50.0f);
				cameraSwitch = 1;
			}
			else if (cameraSwitch == 1) {

				camera.Position = position_model;
				camera.Position.y = camera.Position.y + 15;
				cameraSwitch = 2;
			}
			else if (cameraSwitch == 2) {
				camera.Position = position_modelP2;
				camera.Position.y = camera.Position.y + 15;
				cameraSwitch = 0;
			}
			mPressed = true;
		}
	}
	else {
		mPressed = false;

	}



}

GLuint loadTexture(const char* filename)
{
	// Step1 Create and bind textures
	GLuint textureId = 0;
	glGenTextures(1, &textureId);
	assert(textureId != 0);


	glBindTexture(GL_TEXTURE_2D, textureId);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
	// Step3 Load Textures with dimension data
	int width, height, nrChannels;
	unsigned char* data = stbi_load(filename, &width, &height, &nrChannels, 0);
	if (!data)
	{
		std::cerr << "Error::Texture could not load texture file:" << filename << std::endl;
		return 0;
	}

	// Step4 Upload the texture to the PU
	GLenum format = 0;
	if (nrChannels == 1)
		format = GL_RED;
	else if (nrChannels == 3)
		format = GL_RGB;
	else if (nrChannels == 4)
		format = GL_RGBA;
	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height,
		0, format, GL_UNSIGNED_BYTE, data);

	// Step5 Free resources
	stbi_image_free(data);
	glBindTexture(GL_TEXTURE_2D, 0);
	return textureId;
}

