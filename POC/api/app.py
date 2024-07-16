from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Sample API',
          description='A simple demonstration of a Flask-RESTPlus API')

ns = api.namespace('tasks', description='Task operations')

task_model = api.model('Task', {
    'id': fields.Integer(readonly=True, description='The task unique identifier'),
    'task': fields.String(required=True, description='The task details')
})

tasks = [
    {'id': 1, 'task': 'Build an API'},
    {'id': 2, 'task': '?????'},
    {'id': 3, 'task': 'Profit!'},
]

@ns.route('/')
class TaskList(Resource):
    '''Shows a list of all tasks and lets you POST to add new tasks'''
    @ns.doc('list_tasks')
    @ns.marshal_list_with(task_model)
    def get(self):
        '''List all tasks'''
        return tasks

    @ns.doc('create_task')
    @ns.expect(task_model)
    @ns.marshal_with(task_model, code=201)
    def post(self):
        '''Create a new task'''
        new_task = api.payload
        new_task['id'] = tasks[-1]['id'] + 1 if tasks else 1
        tasks.append(new_task)
        return new_task, 201

@ns.route('/<int:id>')
@ns.response(404, 'Task not found')
@ns.param('id', 'The task identifier')
class Task(Resource):
    '''Show a single task item and lets you delete them'''
    @ns.doc('get_task')
    @ns.marshal_with(task_model)
    def get(self, id):
        '''Fetch a given resource'''
        task = next((task for task in tasks if task['id'] == id), None)
        if task is None:
            api.abort(404)
        return task

    @ns.doc('delete_task')
    @ns.response(204, 'Task deleted')
    def delete(self, id):
        '''Delete a task given its identifier'''
        global tasks
        tasks = [task for task in tasks if task['id'] != id]
        return '', 204

if __name__ == '__main__':
    app.run(debug=True)
